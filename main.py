import re
import string
import pandas as pd
import numpy as np

import streamlit as st

# NLP (Natural Language Toolkit)
import nltk
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from nltk.stem import WordNetLemmatizer #shortening word eg running = run
from nltk.tokenize import word_tokenize #splits sentences into words.
from nltk import pos_tag #part-of-speech tagging (e.g., noun, verb, adj).
from nltk.collocations import BigramCollocationFinder # finds common two-word combinations (bigram features like "not good").
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist
from nltk.text import Text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #feature extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# over-sampling to balance each class
from imblearn.over_sampling import SMOTE

# Transformers (BERT)
import torch
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification # load in trained model
from torch.utils.data import Dataset

#st.write("All keys in session_state:", st.session_state.keys())


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
encoder = LabelEncoder()


bert_label_map = {
    "negative": 0,  # negative
    "neutral": 1,  # neutral
    "positive": 2   # positive
}

# Map POS tags to WordNet tags for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wn.ADJ   # adjective
    elif tag.startswith("V"):
        return wn.VERB  # verb
    elif tag.startswith("N"):
        return wn.NOUN  # noun
    elif tag.startswith("R"):
        return wn.ADV   # adverb
    else:
        return wn.NOUN  # default noun

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\']', ' ', text)
    tokens = word_tokenize(text)  # tokenize
    # remove punctuation + stopwords
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    # POS tagging
    tagged = pos_tag(tokens)
    # Lemmatization with POS
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(tokens)



if "df" not in st.session_state or "tfidf" not in st.session_state:
    df = pd.read_csv("Yelp_Cleaned.csv")
    df["lemma_text"] = df["text"].astype(str).apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # unigrams + bigrams
    # tfidf.fit(df["lemma_text"]).toarray()
    st.session_state.df = df
    st.session_state.tfidf = tfidf

if "X" not in st.session_state or "y" not in st.session_state:
    X = st.session_state.tfidf.fit_transform(st.session_state.df["lemma_text"]).toarray()
    y = encoder.fit_transform(st.session_state.df["label"])
    
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.map_label = encoder.classes_

if "X_train" not in st.session_state:
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.X, st.session_state.y, test_size=0.2, random_state=42
    )
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

# Naive Bayes
if "nb_model" not in st.session_state:
    nb_model = MultinomialNB(alpha=0.01)#, fit_prior=True, class_prior=[0.3, 0.3, 0.4])
    nb_model.fit(st.session_state.X_train, st.session_state.y_train)

    st.session_state.nb_model = nb_model
    st.session_state.y_pred_nb = nb_model.predict(st.session_state.X_test)

# Support Vector Machine
if "svm_model" not in st.session_state:
    svc = LinearSVC(class_weight="balanced",random_state=42)
    svc.fit(st.session_state.X_train, st.session_state.y_train)

    svm_model = CalibratedClassifierCV(svc, cv="prefit") #change to 2 if after can't apply
    svm_model.fit(st.session_state.X_train, st.session_state.y_train)

    st.session_state.svm_model = svm_model
    st.session_state.y_pred_svm = svm_model.predict(st.session_state.X_test)

# BERT
if "bert_pipeline" not in st.session_state:
    #----------------------- IMPORT TRAINED MODEL -----------------------
    # load_path = "./bert_model"
    # tokenizer = AutoTokenizer.from_pretrained(load_path)
    # model = AutoModelForSequenceClassification.from_pretrained(load_path)

    bert_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
        batch_size=32,
        # model=model,
        # tokenizer=tokenizer
    )
    bert_dataset = st.session_state.df["lemma_text"].tolist()   # runned once
    bert_results = bert_pipeline(bert_dataset)

    st.session_state.bert_pipeline = bert_pipeline
    st.session_state.y_pred_bert = [bert_label_map[r["label"]] for r in bert_results]

def _rating(prediction):
    bad, average, good = prediction
    sentiment_score = (bad * 0) + (average * 0.5) + (good * 1)
    return '%.1f' % (1 + (sentiment_score * 4))

def classify_review(model, review):
    processed_review = preprocess_text(review)
    vector = st.session_state.tfidf.transform([processed_review])
    prediction = None
    if model == st.session_state.nb_model:
        prediction = st.session_state.nb_model.predict_proba(vector)[0]
    elif model == st.session_state.svm_model:
        prediction = st.session_state.svm_model.predict_proba(vector)[0]
    elif model == st.session_state.bert_pipeline:
        results = st.session_state.bert_pipeline(review, return_all_scores=True)[0]
        prediction = [0.0, 0.0, 0.0]
        
        for r in results:
            idx = bert_label_map[r["label"]]
            prediction[idx] = round(r["score"], 2)
            
    return [st.session_state.map_label[np.argmax(prediction)], np.round(prediction, 2).tolist()]


def performance_evaluation():
    nb_metrics = {
        "Accuracy": accuracy_score(st.session_state.y_test, st.session_state.y_pred_nb),
        "Precision": precision_score(st.session_state.y_test, st.session_state.y_pred_nb, average="weighted"),
        "Recall": recall_score(st.session_state.y_test, st.session_state.y_pred_nb, average="weighted"),
        "F1 Score": f1_score(st.session_state.y_test, st.session_state.y_pred_nb, average="weighted")
    }
    svm_metrics = {
        "Accuracy": accuracy_score(st.session_state.y_test, st.session_state.y_pred_svm),
        "Precision": precision_score(st.session_state.y_test, st.session_state.y_pred_svm, average="weighted"),
        "Recall": recall_score(st.session_state.y_test, st.session_state.y_pred_svm, average="weighted"),
        "F1 Score": f1_score(st.session_state.y_test, st.session_state.y_pred_svm, average="weighted")
    }
    bert_metrics = {
        "Accuracy": accuracy_score(st.session_state.y, st.session_state.y_pred_bert),
        "Precision": precision_score(st.session_state.y, st.session_state.y_pred_bert, average="weighted"),
        "Recall": recall_score(st.session_state.y, st.session_state.y_pred_bert, average="weighted"),
        "F1 Score": f1_score(st.session_state.y, st.session_state.y_pred_bert, average="weighted")
    }

    return pd.DataFrame([nb_metrics, svm_metrics, bert_metrics],
                       index=["Naïve Bayes", "Support Vector Machine", "BERT"])









st.set_page_config(page_title="Natural Language Processing", layout="wide")
st.title("Restaurant Feedback Classifier")
menu = st.sidebar.radio("Navigation", ["Algorithm Model Call", "Performance Evaluation"])
if menu == "Algorithm Model Call":
    st.header("🧮 Algorithm Model")
    st.markdown("### Select a Model for Prediction")
    st.markdown("Choose one of the following NLP models:")

    st.markdown("""
    <style>
    div.stButton > button {
        height: 2.8em;
        width: 200px;
        font-size: 16px;
        border-radius: 6px;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        color: #111111;
    }
    div.stButton > button:hover {
        background-color: #f3f4f6;
        border-color: #9ca3af;
        color: #111111;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        if st.button("Naïve Bayes"):
            st.session_state["selected_model"] = "Naïve Bayes"
    with col2:
        if st.button("Support Vector Machine"):
            st.session_state["selected_model"] = "Support Vector Machine"
    with col3:
        if st.button("BERT"):
            st.session_state["selected_model"] = "BERT"
    selected_model = None

    selected = st.session_state.get("selected_model", "None") or "Naïve Bayes"
    st.markdown(f"**Selected Model:** {selected}")
    if selected == "Naïve Bayes":
        selected_model = st.session_state.nb_model
    elif selected == "Support Vector Machine":
        selected_model = st.session_state.svm_model
    elif selected == "BERT":
        selected_model = st.session_state.bert_pipeline
    else:
        st.write(f"Please select a valid model.")

    if "review_input_field" not in st.session_state:
        st.session_state.review_input_field = ""

    col_text, col_button = st.columns([5, 1])
    with col_text:
        review = st.text_input("Enter your review here:", value=st.session_state.review_input_field)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        classify_clicked = st.button("Classify")

    if classify_clicked and review and selected_model:
        result, probabilities = classify_review(selected_model, review)
        st.write(f"Classifying review: {result}")
        st.write(f"Rating Score : {_rating(probabilities)}/5⭐")
    elif classify_clicked and (selected_model == None or not review):
        st.write("[Warning] Unable to classify without a model/empty review.")

elif menu == "Performance Evaluation":
    st.header("📊 Performance Evaluation")
    st.markdown("### Models' Performance")
    st.markdown("Performance & Accuracy of 3 NLP Models (Naïve Bayes, Support Vector Machine & BERT)")
    performance = performance_evaluation()
    st.dataframe(performance)