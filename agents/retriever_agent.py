class RetrieverAgent:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)
