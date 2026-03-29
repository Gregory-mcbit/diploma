from typing import List
from langchain_core.documents import Document
from app.rag.client import get_vectorstore
from app.rag.schemas import KnowledgeDocument

COLLECTION_NAME = "knowledge_base"

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
    print(f"[RAG] Added {len(docs)} foundational rules to Knowledge Base.")

def retrieve_rules(query: str, k: int = 3) -> str:
    """
    Search for fixed methodology rules. E.g. Portfolio Builder queries: 
    'What are the rebalancing rules for a conservative profile?'
    """
    db = get_vectorstore(COLLECTION_NAME)
    
    try:
        results = db.similarity_search(query, k=k)
    except Exception:
        return "No specific methodological rules found."
        
    if not results:
        return "No specific methodological rules found."
        
    compiled = "\n".join([f"- [{doc.metadata.get('category', 'RULE')}] {doc.page_content}" for doc in results])
    return compiled
