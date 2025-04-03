import streamlit as st
import requests

st.title("📚 PDF-Based Q&A System")

query = st.text_input("Ask a question based on PDFs:")

if st.button("Search"):
    response = requests.post("http://127.0.0.1:8000/search", json={"query": query})

    if response.status_code == 200:
        results = response.json().get("answers", [])
        for i, answer in enumerate(results):
            st.write(f"*Answer {i+1}:* {answer}")
    else:
        st.error("Error fetching response.")