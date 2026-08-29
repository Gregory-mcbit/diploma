import os
from langchain_openai import ChatOpenAI
from app.observability.logger import get_logger


logger = get_logger(__name__)

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns the standardized GPT-4o-mini LangChain model.
    Using temperature 0 by default for analytical determinism.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for production LLM execution.")
    logger.info("Initializing ChatOpenAI client for model gpt-4o-mini.")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature
        # API key will be automatically loaded from os.environ["OPENAI_API_KEY"] by LangChain
    )
