import crewai
from dotenv import load_dotenv
import os

load_dotenv()  # load .env file

API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure CrewAI to use OpenRouter
crewai.configure(
    default_llm="gpt-3.5-turbo",  # model identifier
    provider="openrouter",
    api_key=API_KEY
)
