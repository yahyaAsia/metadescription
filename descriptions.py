import streamlit as st
import csv
import logging
import requests
import pandas as pd
from io import StringIO, BytesIO
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
from rake_nltk import Rake
from urllib.parse import urlparse
import nltk

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Check for lxml
try:
    import lxml
except ImportError:
    st.error("The 'lxml' package is missing. Please ensure it is included in requirements.txt.")
    st.stop()

# Check for setuptools (previous fix)
try:
    import pkg_resources
except ImportError:
    st.error("The 'setuptools' package is missing. Please ensure it is included in requirements.txt.")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def generate_meta_descriptions(urls):
    results = []
    rake = Rake()
    progress_bar = st.progress(0)
    total_urls = len(urls)

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing URL: {url}")
            # Fetch and parse the page
            parser = HtmlParser.from_url(url, Tokenizer("english"))
            
            # Extract text for keyword analysis
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = ' '.join([p.get_text() for p in soup.find_all('p')])

            # Extract top keywords
            rake.extract_keywords_from_text(page_text)
            keywords = rake.get_ranked_phrases()[:3]
            keyword_str = ', '.join(keywords[:2]) if keywords else ''

            # Summarize content
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
            logger.error(f"Failed to fetch {url}: {e}")
            results.append({
                'url': url,
                'description': 'Error: Could not fetch content.',
                'char_count': 0,
                'keywords': ''
            })
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            results.append({
                'url': url,
                'description': 'Error: Processing failed.',
                'char_count': 0,
                'keywords': ''
            })

        # Update progress bar
        progress_bar.progress((i + 1) / total_urls)

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
else:
    st.info("Please upload a urls.txt file to begin.")

# Instructions
st.markdown("""
### How to Use
1. Create a `urls.txt` file with one URL per line (e.g., `https://example.com/page1`).
2. Upload the file using the uploader above.
3. Click "Generate Meta Descriptions" to process the URLs.
4. Review the results in the table and download the CSV file.
""")
