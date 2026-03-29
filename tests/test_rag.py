import pytest
from app.rag.knowledge_rag import retrieve_rules
from app.rag.news_cache_rag import write_news_cache, check_news_cache
from app.rag.decision_memory_rag import log_decision_to_memory, retrieve_past_mistakes
from app.domain.schemas import CriticReport, CandidatePortfolio

def test_knowledge_retrieval():
    """
    Test that the ChromaDB Knowledge Base accurately pulls institutional Vanguard rules.
    """
    # Ask about conservative profile
    query = "What is the maximum equity exposure for a conservative profile?"
    result = retrieve_rules(query)
    
    # The result should contain the text we initialized (e.g., 'Conservative', '40%')
    assert "Conservative" in result
    assert "40%" in result
    
    # Ask about bear market
    query_bear = "What should I do in a bear market?"
    result_bear = retrieve_rules(query_bear)
    assert "Bear Market" in result_bear
    assert "reduce overall equity exposure" in result_bear

def test_news_cache_flow():
    """
    Test that the News RAG correctly saves and instantly retrieves cached items.
    """
    import uuid
    test_ticker = f"DUMMY_TICKER_{uuid.uuid4()}"
    test_summary = "This is a strictly fake news summary for testing the Cache RAG."
    
    # Ensure it's empty first
    miss = check_news_cache(test_ticker)
    assert miss is None
    
    # Write to cache
    write_news_cache(test_ticker, test_summary, source="pytest")
    
    # Fetch from cache (should be instant hit)
    hit = check_news_cache(test_ticker)
    assert hit is not None
    assert test_summary in hit

import uuid

def test_decision_memory_flow():
    """
    Test that the Critic's Decision Memory correctly stores and retrieves bad portfolios.
    """
    # Use UUIDs so multiple test runs don't pull old dirty DB states
    test_id = str(uuid.uuid4())
    profile = f"Aggressive Testing Profile {test_id}"
    regime = f"Fake Bear Regime {test_id}"
    
    test_weights = {"AAPL": 0.9, "CASH": 0.1}
    test_portfolio = CandidatePortfolio(
        selected_assets=["AAPL"],
        weights=test_weights,
        cash_weight=0.1,
        rationale=["I like Apple."]
    )
    
    test_critic_report = CriticReport(
        issues=["AAPL is 90%, way too concentrated for any decent portfolio."],
        recommended_action="Reduce AAPL to 20%, buy some bonds.",
        verdict="revise_weights"
    )
    
    # Write memory
    log_decision_to_memory(profile, regime, test_portfolio, test_critic_report)
    
    # Retrieve past mistakes
    memory_recall = retrieve_past_mistakes(profile, regime)
    
    assert "AAPL is 90%" in memory_recall
    assert "revise_weights" in memory_recall
