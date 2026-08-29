import uuid

from langchain_core.documents import Document

from app.domain.schemas import CandidatePortfolio, CriticReport, MonitoringAction, MonitoringDecision
from app.rag.decision_memory_rag import (
    log_decision_to_memory,
    log_monitoring_decision,
    retrieve_past_mistakes,
)
from app.rag.knowledge_rag import retrieve_rules
from app.rag.news_cache_rag import check_news_cache, write_news_cache


class FakeVectorStore:
    def __init__(self):
        self.docs: list[Document] = []

    def add_documents(self, docs):
        self.docs.extend(docs)

    def similarity_search(self, query, k=4, filter=None):
        results = list(self.docs)
        if filter:
            for key, expected in filter.items():
                if isinstance(expected, dict) and "$ne" in expected:
                    results = [doc for doc in results if doc.metadata.get(key) != expected["$ne"]]
                else:
                    results = [doc for doc in results if doc.metadata.get(key) == expected]

        query_l = query.lower()
        scored = []
        for doc in results:
            haystack = f"{doc.page_content} {doc.metadata}".lower()
            score = sum(1 for token in query_l.split() if token in haystack)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]


def test_knowledge_retrieval(monkeypatch):
    store = FakeVectorStore()
    store.add_documents(
        [
            Document(
                page_content=(
                    "The Conservative profile prioritizes capital preservation. Maximum equity "
                    "exposure is strictly capped at 40%."
                ),
                metadata={"id": "1", "category": "profile_mapping"},
            ),
            Document(
                page_content=(
                    "In a Bear Market regime, reduce overall equity exposure and increase "
                    "defensive allocations."
                ),
                metadata={"id": "2", "category": "regime_tactical"},
            ),
        ]
    )
    monkeypatch.setattr("app.rag.knowledge_rag.get_vectorstore", lambda _: store)

    result = retrieve_rules("What is the maximum equity exposure for a conservative profile?")
    assert result.matches
    assert "Conservative" in result.matches[0].content
    assert "40%" in result.matches[0].content

    result_bear = retrieve_rules("What should I do in a bear market?")
    assert result_bear.matches
    assert any("Bear Market" in match.content for match in result_bear.matches)
    assert any("reduce overall equity exposure" in match.content for match in result_bear.matches)


def test_news_cache_flow(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr("app.rag.news_cache_rag.get_vectorstore", lambda _: store)

    test_ticker = f"DUMMY_TICKER_{uuid.uuid4()}"
    test_summary = "This is a strictly fake news summary for testing the Cache RAG."

    miss = check_news_cache(test_ticker)
    assert miss.hit is False
    assert miss.freshness_status.value == "unknown"

    write_news_cache(test_ticker, test_summary, source="pytest")
    hit = check_news_cache(test_ticker)
    assert hit.hit is True
    assert hit.payload is not None
    assert test_summary in hit.payload


def test_decision_memory_flow(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr("app.rag.decision_memory_rag.get_vectorstore", lambda _: store)

    test_id = str(uuid.uuid4())
    profile = f"Aggressive Testing Profile {test_id}"
    regime = f"Fake Bear Regime {test_id}"

    test_portfolio = CandidatePortfolio(
        selected_assets=["AAPL"],
        weights={"AAPL": 0.9, "CASH": 0.1},
        cash_weight=0.1,
        rationale=["I like Apple."],
    )

    test_critic_report = CriticReport(
        issues=["AAPL is 90%, way too concentrated for any decent portfolio."],
        recommended_action="Reduce AAPL to 20%, buy some bonds.",
        verdict="revise_weights",
    )

    log_decision_to_memory(profile, regime, test_portfolio, test_critic_report)
    memory_recall = retrieve_past_mistakes(profile, regime)

    assert memory_recall.matches
    assert "AAPL is 90%" in memory_recall.matches[0].content
    assert memory_recall.matches[0].verdict == "revise_weights"


def test_monitoring_decision_memory_flow(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr("app.rag.decision_memory_rag.get_vectorstore", lambda _: store)

    portfolio = CandidatePortfolio(
        selected_assets=["SPY", "TLT"],
        weights={"SPY": 0.6, "TLT": 0.3},
        cash_weight=0.1,
        rationale=["Existing portfolio."],
    )
    decision = MonitoringDecision(
        action=MonitoringAction.reduce_risk,
        reasons=["Risk validation reported hard policy violations."],
        trigger_flags=["risk_threshold_breach"],
        summary="Monitoring decision: reduce_risk.",
    )

    log_monitoring_decision("moderate", "risk_off", portfolio, decision)
    results = store.similarity_search("Monitoring outcome for moderate in risk_off regime", k=1)

    assert results
    assert "reduce_risk" in results[0].page_content
