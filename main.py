import streamlit as st
st.set_page_config(page_title="Natural Language Processing", layout="wide")
menu = st.sidebar.radio("Algorithm Model", ["Naïve Bayes", "Support Vector Machine", "BERT"])
