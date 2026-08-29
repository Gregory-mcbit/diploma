import time
import uuid
from langchain_core.documents import Document
from app.domain.schemas import FreshnessStatus
from app.observability.logger import get_logger
from app.rag.client import get_vectorstore
from app.rag.schemas import NewsCacheLookup

COLLECTION_NAME = "news_cache"
logger = get_logger(__name__)

def write_news_cache(ticker: str, summary_text: str, source: str = "web") -> str:
    db = get_vectorstore(COLLECTION_NAME)
    retrieval_id = str(uuid.uuid4())
    doc = Document(
        page_content=summary_text,
        metadata={
            "id": retrieval_id,
            "ticker": ticker,
            "scraped_at": time.time(),
            "source": source
        }
    )
    db.add_documents([doc])
    logger.info("Cached news for %s.", ticker)
    return retrieval_id

def check_news_cache(ticker: str, max_age_hours: int = 24) -> NewsCacheLookup:
    """
    Searches for recent news for a ticker via Metadata filtering. 
    If it's too old or doesn't exist, returns None (forcing LangGraph to run a fresh scrape).
    """
    db = get_vectorstore(COLLECTION_NAME)
    results = db.similarity_search(
        f"news updates reports about {ticker}", 
        filter={"ticker": ticker},
        k=1
    )
    
    if not results:
        return NewsCacheLookup(
            retrieval_id=str(uuid.uuid4()),
            ticker=ticker,
            hit=False,
            freshness_status=FreshnessStatus.unknown,
        )
        
    best_doc = results[0]
    scraped_at = best_doc.metadata.get("scraped_at", 0)
    age_hours = (time.time() - scraped_at) / 3600.0
    
    if age_hours > max_age_hours:
        logger.info("News cache for %s is stale at %.1f hours.", ticker, age_hours)
        return NewsCacheLookup(
            retrieval_id=str(best_doc.metadata.get("id") or uuid.uuid4()),
            ticker=ticker,
            hit=False,
            freshness_status=FreshnessStatus.stale,
            payload=best_doc.page_content,
            scraped_at=float(scraped_at),
            age_hours=float(age_hours),
            source=best_doc.metadata.get("source"),
        )
        
    logger.info("News cache hit for %s at %.1f hours.", ticker, age_hours)
    return NewsCacheLookup(
        retrieval_id=str(best_doc.metadata.get("id") or uuid.uuid4()),
        ticker=ticker,
        hit=True,
        freshness_status=FreshnessStatus.fresh,
        payload=best_doc.page_content,
        scraped_at=float(scraped_at),
        age_hours=float(age_hours),
        source=best_doc.metadata.get("source"),
    )
