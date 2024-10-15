import streamlit as st
from newspaper import Article
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Function to ensure NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

# Download necessary NLTK data
download_nltk_resources()

# Function to summarize text
def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a CountVectorizer to convert sentences to vectors
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between sentence vectors
    similarity_matrix = cosine_similarity(vectors)

    # Sort sentences by similarity score
    sentence_scores = [sum(similarity_matrix[i]) for i in range(len(sentences))]
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Get the summarized text (top 3 sentences for example)
    summarized_text = " ".join([s[1] for s in ranked_sentences[:3]])
    return summarized_text

# Function to fetch and summarize news article
def summarize_article(url):
    try:
        # Download the article
        article = Article(url)
        article.download()
        article.parse()

        # Ensure the article was successfully downloaded
        if not article.is_parsed:
            return "Error: Article could not be parsed. Please check the URL."

        # Tokenize the article into sentences
        sentences = sent_tokenize(article.text)

        # Create a CountVectorizer to convert sentences to vectors
        vectorizer = CountVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()

        # Calculate cosine similarity between sentence vectors
        similarity_matrix = cosine_similarity(vectors)

        # Sort sentences by similarity score
        sentence_scores = [sum(similarity_matrix[i]) for i in range(len(sentences))]
        ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Get the summarized text (top 3 sentences for example)
        summarized_text = " ".join([s[1] for s in ranked_sentences[:3]])
        return summarized_text
    except Exception as e:
        return f"Error in summarizing the article: {str(e)}. Please check the URL."

# Streamlit App
st.title("TextPrep AI")

# Input text area for text summarization
st.subheader("Summarize Text")
user_text = st.text_area("Enter the text you want to summarize")

# Input field for article URL
st.subheader("Summarize Article URL")
article_url = st.text_input("Enter the URL of the article you want to summarize")

# Button to generate summaries
if st.button("Generate Summary"):
    if user_text:
        st.subheader("Summarized Text")
        summarized_text = summarize_text(user_text)
        st.write(summarized_text)

    if article_url:
        st.subheader("Summarized Article")
        summarized_article = summarize_article(article_url)
        st.write(summarized_article)

# Footer with group details and GitHub link
st.markdown("""
### Group-9: 
- **Members:** Nikhil Thakur, Mukund Tiwari, Rutik Jaybhaye  
- **GitHub Repository:** [View Repository](https://github.com/your-github-repo-url)  
""")
