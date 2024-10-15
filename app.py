import streamlit as st
from newspaper import Article
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Function to ensure NLTK resources are downloaded
def download_nltk_resources():
    nltk.download('punkt', quiet=True)

# Download necessary NLTK data
download_nltk_resources()

# Function to summarize text
def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a CountVectorizer to convert sentences to vectors
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between sentence vectors
    similarity_matrix = cosine_similarity(vectors)

    # Sort sentences by similarity score
    sentence_scores = [sum(similarity_matrix[i]) for i in range(len(sentences))]
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)[:num_sentences]

    # Get the summarized text
    summarized_text = " ".join([s[1] for s in ranked_sentences])
    return summarized_text

# Function to fetch and summarize news article
def summarize_article(url, num_sentences=3):
    try:
        # Download the article
        article = Article(url)
        article.download()
        article.parse()

        # Tokenize the article into sentences
        sentences = sent_tokenize(article.text)

        # Create a CountVectorizer to convert sentences to vectors
        vectorizer = CountVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()

        # Calculate cosine similarity between sentence vectors
        similarity_matrix = cosine_similarity(vectors)

        # Sort sentences by similarity score
        sentence_scores = [sum(similarity_matrix[i]) for i in range(len(sentences))]
        ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)[:num_sentences]

        # Get the summarized text
        summarized_text = " ".join([s[1] for s in ranked_sentences])
        return summarized_text
    except Exception as e:
        return "Error in summarizing the article. Please check the URL."

# Streamlit App
st.title("Text and Article Summarizer")

# Input text area for text summarization
st.subheader("Summarize Text")
user_text = st.text_area("Enter the text you want to summarize")

# Input field for article URL
st.subheader("Summarize Article URL")
article_url = st.text_input("Enter the URL of the article you want to summarize")

# Number of sentences for the summary
num_sentences = st.slider("Number of sentences for summary", 1, 10, 3)

# Button to generate summaries
if st.button("Generate Summary"):
    if user_text:
        st.subheader("Summarized Text")
        summarized_text = summarize_text(user_text, num_sentences)
        st.write(summarized_text)

    if article_url:
        st.subheader("Summarized Article")
        summarized_article = summarize_article(article_url, num_sentences)
        st.write(summarized_article)

# Footer
st.write("### Group-9: Members - Nikhil Thakur, Mukund Tiwari, Rutik Jaybhaye")
