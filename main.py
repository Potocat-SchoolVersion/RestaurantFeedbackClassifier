import streamlit as st
st.set_page_config(page_title="Graduation QR Scanner", layout="wide")
menu = st.sidebar.radio("Navigation", ["🏠 Naïve Bayes", "📝 Support Vector Machine", "🎭 BERT"])
