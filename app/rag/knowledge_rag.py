import uuid
from typing import List
from langchain_core.documents import Document
from app.observability.logger import get_logger
from app.rag.default_knowledge import build_default_knowledge_documents
from app.rag.client import get_vectorstore
from app.rag.schemas import KnowledgeDocument, KnowledgeMatch, KnowledgeRetrievalResult

COLLECTION_NAME = "knowledge_base"
logger = get_logger(__name__)


def _vectorstore_count(db) -> int | None:
    if hasattr(db, "_collection"):
        try:
            return int(db._collection.count())
        except Exception:
            return None
    if hasattr(db, "docs"):
        try:
            return len(db.docs)
        except Exception:
            return None
    return None


def ensure_knowledge_base_seeded() -> None:
    db = get_vectorstore(COLLECTION_NAME)
    count = _vectorstore_count(db)
    if count is not None and count > 0:
        return
    add_knowledge_rules(build_default_knowledge_documents())
    logger.info("Seeded default methodology rules into knowledge base.")

def add_knowledge_rules(rules: List[KnowledgeDocument]) -> None:
    db = get_vectorstore(COLLECTION_NAME)
    docs = []
    for rule in rules:
        doc = Document(
            page_content=rule.content,
            metadata={"id": rule.id, "category": rule.category}
        )
        docs.append(doc)
        
    db.add_documents(docs)
    logger.info("Added %s foundational rules to the knowledge base.", len(docs))

def retrieve_rules(query: str, k: int = 3) -> KnowledgeRetrievalResult:
    """
    Search for fixed methodology rules. E.g. Portfolio Builder queries: 
    'What are the rebalancing rules for a conservative profile?'
    """
    try:
        ensure_knowledge_base_seeded()
        db = get_vectorstore(COLLECTION_NAME)
        results = db.similarity_search(query, k=k)
    except Exception as e:
        raise RuntimeError(f"Knowledge RAG retrieval failed: {e}") from e
        
    matches = [
        KnowledgeMatch(
            id=str(doc.metadata.get("id") or uuid.uuid4()),
            category=str(doc.metadata.get("category", "rule")),
            content=doc.page_content,
        )
        for doc in results
    ]
    return KnowledgeRetrievalResult(
        retrieval_id=str(uuid.uuid4()),
        query=query,
        matches=matches,
    )
