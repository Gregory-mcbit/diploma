from app.rag.default_knowledge import build_default_knowledge_documents
from app.rag.knowledge_rag import add_knowledge_rules

def initialize_knowledge_base():
    docs = build_default_knowledge_documents()
    
    print("[INIT] Injecting Institutional Knowledge Rules into ChromaDB...")
    add_knowledge_rules(docs)
    print("[INIT] Done! The Knowledge RAG is densely populated and ready for Agent retrieval.")

if __name__ == "__main__":
    initialize_knowledge_base()
