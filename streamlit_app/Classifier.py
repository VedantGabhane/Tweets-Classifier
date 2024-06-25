import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import Word2Vec

# Load NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
best_model = joblib.load('logistic_model.pkl')
w2v_model = joblib.load("word2vec.joblib")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    n_words = 0.
    for word in words:
        if word in vocabulary:
            n_words += 1.
            feature_vector = np.add(feature_vector, model.wv[word])
    if n_words:
        feature_vector = np.divide(feature_vector, n_words)
    return feature_vector

def averaged_word_vectorizer(texts, model, num_features):
    vocabulary = set(model.wv.key_to_index)
    features = [average_word_vectors(text.split(), model, vocabulary, num_features) for text in texts]
    return np.array(features)

# Streamlit UI
st.set_page_config(
    page_title="Disaster Tweet Classification",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("Disaster Tweet Classification")

user_input = st.text_area("Enter the tweet text for classification:")

if st.button("Classify"):
    if user_input:
        # Preprocess the input text
        cleaned_text = preprocess_text(user_input)
        
        # TF-IDF Vectorization
        tfidf_features = tfidf_vectorizer.transform([cleaned_text])
        
        # Word2Vec Vectorization
        w2v_features = averaged_word_vectorizer([cleaned_text], w2v_model, 100)
        
        # Combine TF-IDF and Word2Vec Features
        combined_features = np.hstack((tfidf_features.toarray(), w2v_features))
        
        # Predict using the best model
        prediction = best_model.predict(combined_features)
        
        # Display the result with color coding
        if prediction == 1:
            result = "Disaster Tweet"
            st.markdown(
                f'<h2 style="color: red; text-align: center;">{result}</h2>',
                unsafe_allow_html=True
            )
        else:
            result = "Not a Disaster Tweet"
            st.markdown(
                f'<h2 style="color: green; text-align: center;">{result}</h2>',
                unsafe_allow_html=True
            )
    else:
        st.write("Please enter a tweet text to classify.")
st.markdown('</div>', unsafe_allow_html=True)
