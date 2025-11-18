# agents/embedding_agent.py

import os
from chromadb import Client
from chromadb.utils import embedding_functions

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class EmbeddingAgent:
    def __init__(self, persist_directory="vectorstore/chroma_db"):
        self.client = Client()
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )

    def add_documents(self, chunks, metadata=None):
        metadatas = metadata if metadata else [{} for _ in chunks]
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=[str(i) for i in range(len(chunks))]
        )

    def query(self, query_text, n_results=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results["documents"][0]
