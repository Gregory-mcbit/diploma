from app.agents.data import run_data_agent
from app.domain.schemas import MacroData, NewsArticle


def test_data_agent_updates_memory_refs_and_freshness(monkeypatch):
    monkeypatch.setattr("app.agents.data.fetch_market_data", lambda *_args, **_kwargs: "data/cache/test.parquet")
    monkeypatch.setattr(
        "app.agents.data.fetch_macro_snapshot",
        lambda: MacroData(
            values={"vix": 20.0, "yield_spread": 0.2},
            source="test.macro",
        ),
    )
    monkeypatch.setattr(
        "app.agents.data.fetch_latest_news",
        lambda *_args, **_kwargs: (
            {
                "SPY": [
                    NewsArticle(
                        ticker="SPY",
                        source="Reuters",
                        timestamp="2026-04-01T00:00:00+00:00",
                        title="SPY update",
                        summary="Market update.",
                        url="https://example.com/spy",
                        sentiment_tag="positive",
                    )
                ]
            },
            {
                "SPY": type(
                    "Lookup",
                    (),
                    {
                        "retrieval_id": "news-cache-spy",
                        "hit": True,
                        "freshness_status": type("Freshness", (), {"value": "fresh"})(),
                    },
                )()
            },
        ),
    )

    result = run_data_agent({"provenance": {}, "freshness_map": {}, "memory_refs": [], "decision_log": []})

    assert result["freshness_map"]["current_universe"] == "fresh"
    assert result["freshness_map"]["news_digest"] == "fresh"
    assert result["provenance"]["current_universe"].source == "standard_universe_registry"
    assert result["memory_refs"]
    assert result["memory_refs"][0].layer == "news_cache"
