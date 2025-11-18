from agents.document_parser import DocumentParserAgent
from agents.embedding_agent import EmbeddingAgent
from agents.retriever_agent import RetrieverAgent
from agents.answer_agent import AnswerAgent

class RAGPipeline:
    def __init__(self):
        self.parser = DocumentParserAgent()
        self.embedder = EmbeddingAgent()
        self.retriever = RetrieverAgent(self.embedder.vectorstore)
        self.answer_agent = AnswerAgent()

    def ingest_document(self, file_path):
        if file_path.endswith(".pdf"):
            text = self.parser.parse_pdf(file_path)
        else:
            text = self.parser.parse_docx(file_path)
        chunks = self.parser.chunk_text(text)
        self.embedder.add_documents(chunks)
        return len(chunks)

    def answer_query(self, query):
        docs = self.retriever.retrieve(query)
        chunks = [d.page_content for d in docs]
        return self.answer_agent.generate_answer(query, chunks)
