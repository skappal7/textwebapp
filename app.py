import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
import spacy
from nltk.corpus import stopwords
from gensim import corpora, models
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Text preprocessing functions
def preprocess_text(text):
    # Tokenization
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return tokens

# Web scraping function
def scrape_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = [review.text for review in soup.find_all('span', {'class': 'review-text'})]  # Modify according to site structure
    return reviews

# Sentiment analysis function
def sentiment_analysis(text):
    doc = nlp(text)
    sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    for sentence in doc.sents:
        if sentence.sentiment > 0:
            sentiment["positive"] += 1
        elif sentence.sentiment < 0:
            sentiment["negative"] += 1
        else:
            sentiment["neutral"] += 1
    return sentiment

# Main Streamlit app
st.title('Text Analytics Web Application')

# Upload section
st.sidebar.header('Upload Files')
uploaded_files = st.sidebar.file_uploader('Upload your text files, CSV, or Excel files', accept_multiple_files=True)

# URL input section
st.sidebar.header('Input URLs for Web Scraping')
url_input = st.sidebar.text_area('Enter URLs (comma separated)')

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == 'text/plain':
            text = uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            text = ' '.join(df['text_column'].dropna().tolist())  # Modify 'text_column' as needed
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
            text = ' '.join(df['text_column'].dropna().tolist())  # Modify 'text_column' as needed
        st.write(f"Processed text from {uploaded_file.name}:")
        st.write(text[:500])  # Display first 500 characters

# Process URLs for web scraping
if url_input:
    urls = url_input.split(',')
    for url in urls:
        reviews = scrape_reviews(url.strip())
        text = ' '.join(reviews)
        st.write(f"Scraped text from {url.strip()}:")
        st.write(text[:500])  # Display first 500 characters

# Text analysis section
if 'text' in locals():
    # Preprocessing
    tokens = preprocess_text(text)
    token_freq = nltk.FreqDist(tokens)
    
    # Sentiment Analysis
    sentiment = sentiment_analysis(text)
    
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Topic Modeling
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    
    # Visualization
    st.header("Text Analysis Results")
    
    st.subheader("Word Frequency")
    st.bar_chart(token_freq.most_common(10))
    
    st.subheader("Sentiment Analysis")
    st.write(sentiment)
    
    st.subheader("Word Cloud")
    st.image(wordcloud.to_array())
    
    st.subheader("Topic Modeling")
    topics = lda_model.print_topics()
    for topic in topics:
        st.write(topic)

    # Export analysis results
    st.sidebar.header('Export Analysis Results')
    export_format = st.sidebar.selectbox('Select Export Format', ['CSV', JSON, 'Excel'])
    export_button = st.sidebar.button('Export')

    if export_button:
        export_data = {
            "tokens": tokens,
            "token_frequency": token_freq,
            "sentiment": sentiment,
            "topics": topics
        }
        if export_format == 'CSV':
            pd.DataFrame(export_data).to_csv('analysis_results.csv', index=False)
        elif export_format == 'JSON':
            pd.DataFrame(export_data).to_json('analysis_results.json')
        elif export_format == 'Excel':
            pd.DataFrame(export_data).to_excel('analysis_results.xlsx', index=False)
        st.sidebar.write('Analysis results exported successfully.')

# Comparative analysis section
st.header("Comparative Analysis")
if len(uploaded_files) > 1 or url_input:
    st.write("Comparative analysis between multiple documents or sources will be shown here.")
