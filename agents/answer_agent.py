from crewai import Agent

class AnswerAgent:
    def __init__(self):
        """
        This agent generates answers based on document chunks using CrewAI + OpenRouter
        """
        self.agent = Agent(
            name="Answer Agent",
            role="Generate context-aware answers based on retrieved document chunks",
            goal="Answer user questions using relevant document snippets",
            llm="gpt-4o-mini"  # CrewAI will use OpenRouter API
        )

    def generate_answer(self, query, context_chunks):
        """
        Generate an answer for the user's question using context chunks
        """
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "You are an expert assistant. Use ONLY the following context to answer the question:\n"
            f"{context_text}\n\n"
            f"Question: {query}"
        )
        return self.agent.run({"query": prompt})
