import time
from typing import Optional
from langchain_core.documents import Document
from app.rag.client import get_vectorstore

COLLECTION_NAME = "news_cache"

def write_news_cache(ticker: str, summary_text: str, source: str = "web") -> None:
    db = get_vectorstore(COLLECTION_NAME)
    
    doc = Document(
        page_content=summary_text,
        metadata={
            "ticker": ticker, 
            "scraped_at": time.time(),
            "source": source
        }
    )
    db.add_documents([doc])
    print(f"[RAG] Cached news for {ticker}.")

def check_news_cache(ticker: str, max_age_hours: int = 24) -> Optional[str]:
    """
    Searches for recent news for a ticker via Metadata filtering. 
    If it's too old or doesn't exist, returns None (forcing LangGraph to run a fresh scrape).
    """
    db = get_vectorstore(COLLECTION_NAME)
    
    try:
        results = db.similarity_search(
            f"news updates reports about {ticker}", 
            filter={"ticker": ticker},
            k=1
        )
    except Exception:
        # If collection is literally completely empty, chroma throws validation exceptions sometimes on empty metadata
        return None
    
    if not results:
        return None
        
    best_doc = results[0]
    scraped_at = best_doc.metadata.get("scraped_at", 0)
    age_hours = (time.time() - scraped_at) / 3600.0
    
    if age_hours > max_age_hours:
        print(f"[RAG Cache Miss] Cache for {ticker} is stale ({age_hours:.1f} hrs). Returning None.")
        return None
        
    print(f"[RAG Cache Hit] Found fresh context for {ticker} (age: {age_hours:.1f} hrs).")
    return best_doc.page_content
