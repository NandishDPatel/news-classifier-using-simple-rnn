import streamlit as st
import numpy as np

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = load_model("news_classification_simple_rnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_SEQ_LENGTH = 90

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def preprocess_text(text):
    text = lemmatize_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post"
    )
    return padded

st.set_page_config(page_title="News Classifier", layout="centered")

st.title("📰 News Classification App")

news_heading = st.text_area("Paste your News Heading ",height=100)
news_article = st.text_area("Paste your News Article Content",height=250)

if st.button("Classify the News"):
    if news_heading.strip() == "" or news_article.strip() == "":
        st.warning("Please copy paste the News Heading and Article to classify.")
    else:
        user_input = news_heading + news_article
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_labels = ["World", "Sports", "Business", "Science & Technology"]
        st.success(f"Predicted News Category: **{class_labels[predicted_class]}**")
else:
    st.info("Copy paste some news text and click Classify News button.")
