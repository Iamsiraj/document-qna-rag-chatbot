import streamlit as st
from rag_pipeline import RAGPipeline

st.title("PDF / Document Q&A Chatbot")

pipeline = RAGPipeline()

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
if uploaded_file:
    bytes_data = uploaded_file.read()
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(bytes_data)
    num_chunks = pipeline.ingest_document(f"temp_{uploaded_file.name}")
    st.success(f"Document ingested with {num_chunks} chunks!")

query = st.text_input("Ask a question about the documents:")
if st.button("Get Answer") and query:
    answer = pipeline.answer_query(query)
    st.write("**Answer:**")
    st.write(answer)
