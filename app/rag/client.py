import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.observability.logger import get_logger

CHROMA_DIR = "data/chroma_db/"
os.makedirs(CHROMA_DIR, exist_ok=True)
logger = get_logger(__name__)

# Singleton to prevent reloading the model weights
_embedding_function = None

def get_embeddings():
    """
    Returns the HuggingFace local embedding model. 
    It will download the model weights (~80MB) from HF hub on the first run.
    """
    global _embedding_function
    if _embedding_function is None:
        logger.info("Initializing HuggingFace embeddings model all-MiniLM-L6-v2.")
        try:
            _embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace embeddings: {e}") from e
    return _embedding_function

def get_vectorstore(collection_name: str) -> Chroma:
    """Returns a LangChain Chroma vectorstore for the given collection."""
    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vectorstore '{collection_name}': {e}") from e
