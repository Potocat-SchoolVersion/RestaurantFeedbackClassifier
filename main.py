import streamlit as st
st.set_page_config(page_title="Graduation QR Scanner", layout="wide")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📝 Registration", "🎭 Stage Scanning", "👥 Management", "📊 Reports"])
