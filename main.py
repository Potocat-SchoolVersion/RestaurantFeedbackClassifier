import streamlit as st
st.set_page_config(page_title="Graduation QR Scanner", layout="wide")
menu = st.sidebar.radio("Jump from Arena", ["🏠 Home", "📝 Registration", "🎭 Stage Scanning", "👥 Management", "📊 Reports"])
