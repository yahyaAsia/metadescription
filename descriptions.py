import streamlit as st
import logging
import requests
import pandas as pd
from io import StringIO, BytesIO
from bs4 import BeautifulSoup
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from rake_nltk import Rake
from urllib.parse import urlparse
import nltk
import time
import traceback

# Set up logging with detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK punkt found")
except LookupError:
    logger.info("Downloading NLTK punkt")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
    logger.info("NLTK stopwords found")
except LookupError:
    logger.info("Downloading NLTK stopwords")
    nltk.download('stopwords')

# Streamlit app title
st.title("Bulk Meta Description Generator")
st.markdown("""
Upload a `urls.txt` file with one URL per line to generate SEO-friendly meta descriptions (under 155 characters).
The app summarizes page content, extracts keywords, and exports results to a CSV file.
""")

# File uploader
uploaded_file = st.file_uploader("Upload urls.txt", type=["txt"])

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []

def fetch_page_content(url, max_retries=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            logger.debug(f"Successfully fetched {url}: {response.status_code}")
            return response.text, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                return None, str(e)
        except Exception as e:
            logger.error(f"Unexpected fetch error for {url}: {e}")
            return None, str(e)

def generate_meta_descriptions(urls):
    results = []
    progress_bar = st.progress(0)
    total_urls = len(urls)

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing URL: {url}")
            # Fetch page content
            html_content, fetch_error = fetch_page_content(url)
            if not html_content:
                raise ValueError(f"Failed to fetch page content: {fetch_error}")

            # Parse with BeautifulSoup
            logger.debug(f"Parsing HTML for {url}")
            soup = BeautifulSoup(html_content, 'html.parser')
            page_text = ' '.join([elem.get_text().strip() for elem in soup.find_all(['p', 'div', 'article']) if elem.get_text().strip()])
            if not page_text or len(page_text) < 100:
                raise ValueError(f"Insufficient text content found (length: {len(page_text)})")

            logger.debug(f"Extracted {len(page_text)} characters of text for {url}")

            # Use PlaintextParser for summarization
            logger.debug(f"Creating PlaintextParser for {url}")
            parser = PlaintextParser.from_string(page_text, Tokenizer("english"))

            # Extract top keywords
            logger.debug(f"Extracting keywords for {url}")
            rake = Rake()
            rake.extract_keywords_from_text(page_text)
            keywords = rake.get_ranked_phrases()[:3]
            keyword_str = ', '.join(keywords[:2]) if keywords else ''
            logger.debug(f"Keywords for {url}: {keyword_str}")

            # Summarize content
            logger.debug(f"Summarizing content for {url}")
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            summary = summarizer(parser.document, 2)
            description = " ".join([sentence._text for sentence in summary]).strip()

            # Clean and truncate description
            if not description:
                description = f"Explore {keyword_str} at {urlparse(url).netloc}." if keyword_str else f"Visit {urlparse(url).netloc} for more info."
            if len(description) > 155:
                description = description[:152] + '...'
            elif len(description) < 50 and keyword_str:
                description = f"{description} Learn about {keyword_str}."

            # Append results
            results.append({
                'url': url,
                'description': description,
                'char_count': len(description),
                'keywords': keyword_str
            })
            logger.info(f"Generated description for {url}: {description[:50]}...")

        except requests.exceptions.RequestException as e:
            logger.error(f"Fetch error for {url}: {e}\n{traceback.format_exc()}")
            results.append({
                'url': url,
                'description': f"Error: Failed to fetch content - {str(e)}",
                'char_count': 0,
                'keywords': ''
            })
        except ValueError as e:
            logger.error(f"Content error for {url}: {e}\n{traceback.format_exc()}")
            results.append({
                'url': url,
                'description': f"Error: {str(e)}",
                'char_count': 0,
                'keywords': ''
            })
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}\n{traceback.format_exc()}")
            results.append({
                'url': url,
                'description': f"Error: Processing failed - {str(e)}",
                'char_count': 0,
                'keywords': ''
            })

        # Update progress bar
        progress_bar.progress((i + 1) / total_urls)
        time.sleep(2)  # Avoid rate-limiting

    return results

# Process the uploaded file
if uploaded_file is not None:
    try:
        # Read the uploaded file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        urls = [line.strip() for line in stringio if line.strip()]
        st.success(f"Imported {len(urls)} URLs from {uploaded_file.name}")

        # Button to start processing
        if st.button("Generate Meta Descriptions"):
            with st.spinner("Generating meta descriptions..."):
                st.session_state.results = generate_meta_descriptions(urls)

            # Display results
            if st.session_state.results:
                df = pd.DataFrame(st.session_state.results)
                st.subheader("Results Preview")
                st.dataframe(df)

                # Convert results to CSV for download
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode('utf-8')

                # Download button
                st.download_button(
                    label="Download meta_descriptions.csv",
                    data=csv_bytes,
                    file_name="meta_descriptions.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Error reading file: {e}")
        logger.error(f"File reading error: {e}\n{traceback.format_exc()}")
else:
    st.info("Please upload a urls.txt file to begin.")

# Instructions
st.markdown("""
### How to Use
1. Create a `urls.txt` file with one URL per line (e.g., `https://example.com/page1`).
2. Upload the file using the uploader above.
3. Click "Generate Meta Descriptions" to process the URLs.
4. Review the results in the table and download the CSV file.

### Debugging Tips
- Check the logs in Streamlit Cloud ("Manage app") for detailed error messages.
- Ensure URLs are accessible and not blocked by the target website.
- Contact support if errors persist, providing the CSV output and logs.
""")
