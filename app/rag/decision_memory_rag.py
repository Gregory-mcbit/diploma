import json
from typing import List
from langchain_core.documents import Document
from app.rag.client import get_vectorstore
from app.domain.schemas import CriticReport, CandidatePortfolio

COLLECTION_NAME = "decision_memory"

def log_decision_to_memory(
    profile_type: str, 
    regime: str, 
    portfolio: CandidatePortfolio, 
    critic_report: CriticReport
) -> None:
    """
    Saves the full context of a critic's verdict into RAG VectorStore.
    This prevents the Critic Agent from repeating identical mistakes in the future.
    """
    db = get_vectorstore(COLLECTION_NAME)
    
    # We embed the critical issues so similar future states trigger this exact memory
    content_to_embed = f"Proposed Portfolio for {profile_type} in {regime} regime. Critic Issues: {', '.join(critic_report.issues)}. Critic Recommended Action: {critic_report.recommended_action}"
    
    state_json = json.dumps({
        "weights": portfolio.weights,
        "rationale": portfolio.rationale
    })
    
    doc = Document(
        page_content=content_to_embed,
        metadata={
            "profile": profile_type,
            "regime": regime,
            "verdict": critic_report.verdict.value,
            "portfolio_json": state_json
        }
    )
    
    db.add_documents([doc])
    print("[RAG Memory] Saved Critic decision to persistent memory.")

def retrieve_past_mistakes(profile_type: str, regime: str) -> str:
    """
    The Critic Agent calls this BEFORE grading the current portfolio. 
    It pulls up the gravestones of past rejected portfolios so it knows what not to accept.
    """
    db = get_vectorstore(COLLECTION_NAME)
    query = f"Past portfolio rejections and mistakes for {profile_type} profile in {regime} regime"
    
    try:
        # We only want to look at things that were explicitly BAD (not 'approve')
        results = db.similarity_search(
            query, 
            k=2,
            filter={"verdict": {"$ne": "approve"}}
        )
    except Exception:
        return "No relevant past mistakes found."
    
    if not results:
        return "No relevant past mistakes found."
        
    context = "Past Critic Feedback (DO NOT REPEAT THESE MISTAKES):\n"
    for r in results:
        context += f"- Verdict: {r.metadata.get('verdict')} | Context: {r.page_content}\n"
    return context
