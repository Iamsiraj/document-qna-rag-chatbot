import streamlit as st
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
import os

# Set your OpenRouter or OpenAI API key
OPENROUTER_API_KEY = ""

# Simple function to load and split PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

vector_db = None

st.title("PDF QnA Chatbot (No CrewAI, No Chroma)")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        docs.extend(load_pdf(file_path))
        os.remove(file_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    st.success("PDFs processed and stored in vector DB.")

question = st.text_input("Ask a question about your PDFs:")

if question and vector_db:
    # Retrieve relevant context
    results = vector_db.similarity_search(question, k=3)
    context = "\n".join([r.page_content for r in results])

    # Query LLM (OpenRouter)
    llm = OpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        model_name="gpt-3.5-turbo",
        openai_api_base="https://openrouter.ai/api/v1/chat/completions"
    )
    prompt = f"Context from PDFs:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = llm(prompt)
    st.write(answer)
elif question:
    st.warning("Please upload and process PDFs first.")
