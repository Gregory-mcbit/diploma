import pandas as pd

from app.domain.schemas import Constraints, MacroData, NewsArticle, RiskProfile, InvestorProfile
from app.agents.scoring import run_scoring_agent


def test_scoring_agent_outputs_fundamentals_features_and_scores(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.5 for i in range(len(dates))],
            "TLT": [90 + i * 0.1 for i in range(len(dates))],
        },
        index=dates,
    )
    parquet_path = tmp_path / "scores.parquet"
    prices.to_parquet(parquet_path)

    monkeypatch.setattr(
        "app.agents.scoring.run_ml_scoring_pipeline",
        lambda parquet_path, universe, macro: {"SPY": 0.05, "TLT": 0.02},
    )
    monkeypatch.setattr(
        "app.agents.scoring.fetch_fundamentals",
        lambda universe: {
            "SPY": type("Snapshot", (), {"metrics": {"pe": 20.0, "forward_pe": 18.0, "price_to_book": 4.0, "roe": 0.2, "net_margin": 0.15, "debt_to_equity": 0.8}})(),
            "TLT": type("Snapshot", (), {"metrics": {"pe": None, "forward_pe": None, "price_to_book": None, "roe": None, "net_margin": None, "debt_to_equity": None}})(),
        },
    )

    state = {
        "market_data_pointer": str(parquet_path),
        "current_universe": ["SPY", "TLT"],
        "macro_data": MacroData(
            values={
                "vix": 18.0,
                "yield_10y": 4.1,
                "yield_3m": 3.9,
                "yield_spread": 0.2,
                "credit_spread": 0.01,
                "usd_strength": 0.02,
                "oil_mom_1m": 0.03,
                "spy_tlt_ratio": 0.01,
            },
            source="test",
        ),
        "news_articles": {
            "SPY": [
                NewsArticle(
                    ticker="SPY",
                    source="Reuters",
                    timestamp="2024-04-01T00:00:00+00:00",
                    title="SPY gains",
                    summary="Strong quarter.",
                    url="https://example.com/spy",
                    sentiment_tag="positive",
                )
            ],
            "TLT": [],
        },
        "provenance": {},
        "freshness_map": {},
        "decision_log": [],
        "profile": InvestorProfile(
            risk_profile=RiskProfile.moderate,
            horizon_years=10,
            target="Balanced growth",
            income_preference="growth",
            constraints=Constraints(
                max_asset_weight=0.25,
                max_sector_weight=0.35,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.20,
                min_cash_weight=0.02,
                max_correlation_threshold=0.85,
            ),
        ),
    }

    result = run_scoring_agent(state)

    assert set(result["fundamentals"].keys()) == {"SPY", "TLT"}
    assert set(result["features"].keys()) == {"SPY", "TLT"}
    assert set(result["asset_scores"].keys()) == {"SPY", "TLT"}
    assert result["asset_scores"]["SPY"].overall_score > 0.05
    assert result["asset_scores"]["SPY"].factors.extra_factors["base_model_score"] == 0.05
    assert result["asset_scores"]["SPY"].confidence > 0.7
    assert result["freshness_map"]["fundamentals"] == "fresh"
