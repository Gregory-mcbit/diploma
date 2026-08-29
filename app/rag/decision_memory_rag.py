import json
import uuid
from typing import List
from langchain_core.documents import Document
from app.observability.logger import get_logger
from app.rag.client import get_vectorstore
from app.domain.schemas import CriticReport, CandidatePortfolio, MonitoringDecision
from app.rag.schemas import DecisionMemoryMatch, DecisionMemoryQueryResult

COLLECTION_NAME = "decision_memory"
logger = get_logger(__name__)

def log_decision_to_memory(
    profile_type: str, 
    regime: str, 
    portfolio: CandidatePortfolio, 
    critic_report: CriticReport
) -> str:
    """
    Saves the full context of a critic's verdict into RAG VectorStore.
    This prevents the Critic Agent from repeating identical mistakes in the future.
    """
    # We embed the critical issues so similar future states trigger this exact memory
    content_to_embed = f"Proposed Portfolio for {profile_type} in {regime} regime. Critic Issues: {', '.join(critic_report.issues)}. Critic Recommended Action: {critic_report.recommended_action}"
    
    state_json = json.dumps({
        "weights": portfolio.weights,
        "rationale": portfolio.rationale
    })

    db = get_vectorstore(COLLECTION_NAME)
    retrieval_id = str(uuid.uuid4())
    doc = Document(
        page_content=content_to_embed,
        metadata={
            "id": retrieval_id,
            "profile": profile_type,
            "regime": regime,
            "verdict": critic_report.verdict.value,
            "portfolio_json": state_json
        }
    )
    db.add_documents([doc])
    logger.info("Saved critic decision to persistent memory.")
    return retrieval_id

def retrieve_past_mistakes(profile_type: str, regime: str) -> DecisionMemoryQueryResult:
    """
    The Critic Agent calls this BEFORE grading the current portfolio. 
    It pulls up the gravestones of past rejected portfolios so it knows what not to accept.
    """
    query = f"Past portfolio rejections and mistakes for {profile_type} profile in {regime} regime"
    
    db = get_vectorstore(COLLECTION_NAME)
    # We only want to look at things that were explicitly BAD (not 'approve')
    results = db.similarity_search(
        query, 
        k=2,
        filter={"verdict": {"$ne": "approve"}}
    )
    
    matches = [
        DecisionMemoryMatch(
            id=str(result.metadata.get("id") or uuid.uuid4()),
            profile_type=str(result.metadata.get("profile", profile_type)),
            regime=str(result.metadata.get("regime", regime)),
            verdict=str(result.metadata.get("verdict", "")),
            content=result.page_content,
            portfolio_json=result.metadata.get("portfolio_json"),
        )
        for result in results
    ]
    return DecisionMemoryQueryResult(
        retrieval_id=str(uuid.uuid4()),
        query=query,
        matches=matches,
    )


def log_monitoring_decision(
    profile_type: str,
    regime: str,
    portfolio: CandidatePortfolio,
    monitoring_decision: MonitoringDecision,
) -> str:
    content_to_embed = (
        f"Monitoring outcome for {profile_type} in {regime} regime. "
        f"Action: {monitoring_decision.action.value}. "
        f"Reasons: {', '.join(monitoring_decision.reasons)}."
    )
    state_json = json.dumps(
        {
            "weights": portfolio.weights,
            "action": monitoring_decision.action.value,
            "trigger_flags": monitoring_decision.trigger_flags,
        }
    )

    db = get_vectorstore(COLLECTION_NAME)
    retrieval_id = str(uuid.uuid4())
    doc = Document(
        page_content=content_to_embed,
        metadata={
            "id": retrieval_id,
            "profile": profile_type,
            "regime": regime,
            "verdict": f"monitoring:{monitoring_decision.action.value}",
            "portfolio_json": state_json,
        }
    )
    db.add_documents([doc])
    logger.info("Saved monitoring decision to persistent memory.")
    return retrieval_id
