from app.domain.schemas import (
    BacktestResult,
    CandidatePortfolio,
    Constraints,
    CriticReport,
    CriticVerdict,
    FinalRecommendation,
    InvestorProfile,
    MacroData,
    RegimeReport,
    RegimeType,
    RiskProfile,
    RiskReport,
)
from app.graph.pipeline import build_investment_graph, build_monitoring_graph
from app.rag.schemas import DecisionMemoryQueryResult


def test_portfolio_correlation_screen_prunes_weaker_asset():
    import numpy as np
    import pandas as pd

    from app.agents.portfolio import _prune_correlated_assets
    from app.domain.schemas import AssetScore, FactorScores

    dates = pd.date_range("2024-01-01", periods=260)
    np.random.seed(21)
    base_returns = np.random.normal(0.0007, 0.01, len(dates))
    qqq_returns = base_returns + np.random.normal(0.0, 0.0005, len(dates))
    tlt_returns = np.random.normal(0.0002, 0.005, len(dates))
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.cumprod(1 + base_returns),
            "QQQ": 100 * np.cumprod(1 + qqq_returns),
            "TLT": 100 * np.cumprod(1 + tlt_returns),
        },
        index=dates,
    )

    asset_scores = {
        "SPY": AssetScore(
            asset_ticker="SPY",
            factors=FactorScores(momentum=0.3, volatility=-0.1, quality=0.4),
            overall_score=0.04,
            confidence=0.9,
        ),
        "QQQ": AssetScore(
            asset_ticker="QQQ",
            factors=FactorScores(momentum=0.4, volatility=-0.1, quality=0.4),
            overall_score=0.06,
            confidence=0.9,
        ),
        "TLT": AssetScore(
            asset_ticker="TLT",
            factors=FactorScores(momentum=0.1, volatility=-0.2, quality=0.6),
            overall_score=0.03,
            confidence=0.9,
        ),
    }

    filtered_scores, exclusion_reasons = _prune_correlated_assets(
        asset_scores=asset_scores,
        price_df=prices,
        max_correlation_threshold=0.85,
    )

    assert "QQQ" in filtered_scores
    assert "TLT" in filtered_scores
    assert "SPY" not in filtered_scores
    assert "SPY" in exclusion_reasons
    assert "слишком сильно коррелировал" in exclusion_reasons["SPY"]


def test_langgraph_pipeline_with_revision_loop(monkeypatch):
    def stub_profile_agent(state):
        return {
            "profile": InvestorProfile(
                risk_profile=RiskProfile.moderate,
                horizon_years=10,
                target="Long-term balanced growth.",
                constraints=Constraints(
                    max_asset_weight=0.25,
                    max_sector_weight=0.35,
                    allowed_asset_classes=["stocks", "bonds", "commodities"],
                    forbidden_assets=[],
                    max_drawdown_tolerance=0.20,
                    min_cash_weight=0.02,
                ),
            )
        }

    def stub_data_agent(state):
        return {
            "current_universe": ["SPY", "TLT", "GLD"],
            "market_data_pointer": "dummy.parquet",
            "news_articles": {},
            "news_digest": {"article_count": 0, "tickers": [], "sentiment_totals": {}, "by_ticker": {}},
        }

    def stub_scoring_agent(state):
        return {"asset_scores": {"SPY": 0.06, "TLT": 0.03, "GLD": 0.02}}

    def stub_regime_agent(state):
        return {
            "market_regime": RegimeReport(
                current_regime=RegimeType.risk_off,
                confidence=0.8,
                drivers=["High VIX", "Negative momentum", "Defensive rotation"],
                is_risk_off=True,
            )
        }

    def stub_backtest_node(state):
        return {
            "backtest_result": BacktestResult(
                portfolio_total_return=0.12,
                benchmark_total_return=0.10,
                equal_weight_total_return=0.08,
                portfolio_volatility=0.14,
                portfolio_max_drawdown=0.09,
                turnover=0.15,
                observations=252,
            )
        }

    def stub_portfolio_agent(state):
        revision_count = state.get("revision_count", 0)
        if revision_count == 0:
            weights = {"SPY": 0.70, "TLT": 0.20}
        else:
            weights = {"SPY": 0.25, "TLT": 0.55, "GLD": 0.10}

        return {
            "proposed_portfolio": CandidatePortfolio(
                selected_assets=list(weights.keys()),
                weights=weights,
                cash_weight=max(0.0, 1.0 - sum(weights.values())),
                rationale=["Deterministic test portfolio."],
            )
        }

    def stub_risk_agent(state):
        portfolio = state["proposed_portfolio"]
        violations = []
        if portfolio.weights.get("SPY", 0.0) > 0.25:
            violations.append("Hard constraint breach: SPY weight exceeds max 25%.")
        return {
            "risk_report": RiskReport(
                portfolio_volatility=0.12,
                max_drawdown_estimate=0.08,
                avg_correlation=0.35,
                violations=violations,
                warnings=[],
                fit_to_profile="acceptable" if not violations else "violation",
            )
        }

    def stub_critic_agent(state):
        revision_count = state.get("revision_count", 0)
        if revision_count == 0:
            report = CriticReport(
                verdict=CriticVerdict.revise_weights,
                issues=["Equity concentration too high for this regime."],
                recommended_action="Reduce SPY and increase defensive sleeves.",
            )
        else:
            report = CriticReport(
                verdict=CriticVerdict.approve,
                issues=[],
                recommended_action="Approved.",
            )
        history = list(state.get("critic_history", []))
        history.append(report)
        return {"critic_report": report, "critic_history": history}

    def stub_explainability_agent(state):
        portfolio = state["proposed_portfolio"]
        return {
            "final_report": FinalRecommendation(
                portfolio=portfolio,
                executive_summary="Graph test final report.",
                regime_context="Risk-off regime favored defensive exposure.",
                risk_disclaimer="Test disclaimer.",
                inclusion_reasons={"SPY": "Included as a core risk asset."},
                exclusion_reasons={"QQQ": "Excluded due to policy concentration controls."},
                uncertainty_notes=["Signal dispersion is moderate."],
                policy_summary=["Risk-off rules tightened max asset weight."],
                memory_comparison="Compared against prior rejected patterns.",
            )
        }

    monkeypatch.setattr("app.graph.pipeline.run_profile_agent", stub_profile_agent)
    monkeypatch.setattr("app.graph.pipeline.run_data_agent", stub_data_agent)
    monkeypatch.setattr("app.graph.pipeline.run_scoring_agent", stub_scoring_agent)
    monkeypatch.setattr("app.graph.pipeline.run_regime_agent", stub_regime_agent)
    monkeypatch.setattr("app.graph.pipeline.run_backtest_node", stub_backtest_node)
    monkeypatch.setattr("app.graph.pipeline.run_portfolio_agent", stub_portfolio_agent)
    monkeypatch.setattr("app.graph.pipeline.run_risk_agent", stub_risk_agent)
    monkeypatch.setattr("app.graph.pipeline.run_critic_agent", stub_critic_agent)
    monkeypatch.setattr("app.graph.pipeline.run_explainability_agent", stub_explainability_agent)

    graph = build_investment_graph()
    result = graph.invoke(
        {
            "user_query": "Balanced portfolio for the next 10 years.",
            "request_id": "req-graph-test",
            "correlation_id": "corr-graph-test",
            "revision_count": 0,
            "critic_history": [],
            "provenance": {},
            "decision_log": [],
            "freshness_map": {},
            "memory_refs": [],
            "trace_log": [],
            "fundamentals": {},
            "macro_data": None,
            "features": {},
            "news_articles": {},
            "news_digest": None,
            "backtest_result": None,
        }
    )

    assert result["critic_report"].verdict == CriticVerdict.approve
    assert result["revision_count"] == 1
    assert len(result["critic_history"]) == 2
    assert result["final_report"].portfolio.cash_weight >= 0.0
    assert set(result["final_report"].portfolio.selected_assets) == {"SPY", "TLT", "GLD"}
    assert "effective_policy" in result
    assert "provenance" in result
    assert "effective_policy" in result["provenance"]
    assert len(result["decision_log"]) >= 1
    assert result["backtest_result"].portfolio_total_return == 0.12
    assert len(result["trace_log"]) >= 2
    assert result["final_report"].policy_summary
    assert result["final_report"].uncertainty_notes
    assert result["freshness_map"]["effective_policy"] == "fresh"
    assert result["freshness_map"]["active_portfolio"] == "fresh"


def test_monitoring_graph_generates_decision(monkeypatch, tmp_path):
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=260)
    np.random.seed(11)
    returns = pd.DataFrame(
        {
            "SPY": np.random.normal(0.0005, 0.012, len(dates)),
            "TLT": np.random.normal(0.0002, 0.006, len(dates)),
            "GLD": np.random.normal(0.0003, 0.007, len(dates)),
        },
        index=dates,
    )
    prices = 100 * np.cumprod(1 + returns)
    parquet_path = tmp_path / "monitoring_prices.parquet"
    prices.to_parquet(parquet_path)

    def stub_data_agent(state):
        return {
            "current_universe": ["SPY", "TLT", "GLD"],
            "market_data_pointer": str(parquet_path),
            "news_articles": {},
            "news_digest": {"article_count": 0, "tickers": [], "sentiment_totals": {}, "by_ticker": {}},
        }

    def stub_scoring_agent(state):
        return {
            "asset_scores": {"SPY": -0.02, "TLT": 0.03, "GLD": 0.01},
            "macro_data": {
                "values": {
                    "vix": 31.0,
                    "yield_10y": 4.3,
                    "yield_3m": 4.9,
                    "yield_spread": -0.6,
                    "credit_spread": -0.02,
                    "usd_strength": 0.01,
                    "oil_mom_1m": -0.03,
                    "spy_tlt_ratio": -0.02,
                },
                "source": "test",
            },
        }

    def stub_regime_agent(state):
        return {
            "market_regime": RegimeReport(
                current_regime=RegimeType.risk_off,
                confidence=0.85,
                drivers=["Elevated volatility", "Negative curve spread", "Weak equity tone"],
                is_risk_off=True,
            )
        }

    def stub_risk_agent(state):
        return {
            "risk_report": RiskReport(
                portfolio_volatility=0.18,
                max_drawdown_estimate=0.12,
                avg_correlation=0.42,
                violations=["Hard constraint breach: SPY weight exceeds max 25%."],
                warnings=[],
                fit_to_profile="violation",
            )
        }

    monkeypatch.setattr("app.graph.pipeline.run_data_agent", stub_data_agent)
    monkeypatch.setattr("app.graph.pipeline.run_scoring_agent", stub_scoring_agent)
    monkeypatch.setattr("app.graph.pipeline.run_regime_agent", stub_regime_agent)
    monkeypatch.setattr("app.graph.pipeline.run_risk_agent", stub_risk_agent)
    monkeypatch.setattr(
        "app.agents.monitoring.retrieve_past_mistakes",
        lambda *_args, **_kwargs: DecisionMemoryQueryResult(
            retrieval_id="memory-test",
            query="test",
            matches=[],
        ),
    )
    monkeypatch.setattr("app.agents.monitoring.log_monitoring_decision", lambda *_args, **_kwargs: None)

    graph = build_monitoring_graph()
    result = graph.invoke(
        {
            "profile": InvestorProfile(
                risk_profile=RiskProfile.moderate,
                horizon_years=10,
                target="Balanced growth",
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
            "active_portfolio": CandidatePortfolio(
                selected_assets=["SPY", "TLT", "GLD"],
                weights={"SPY": 0.55, "TLT": 0.30, "GLD": 0.10},
                cash_weight=0.05,
                rationale=["Existing active portfolio."],
            ),
            "request_id": "req-monitor-graph-test",
            "correlation_id": "corr-monitor-graph-test",
            "provenance": {},
            "decision_log": [],
            "freshness_map": {},
            "memory_refs": [],
            "trace_log": [],
            "fundamentals": {},
            "macro_data": None,
            "features": {},
            "news_articles": {},
            "news_digest": None,
            "backtest_result": None,
        }
    )

    assert result["monitoring_decision"].action.value in {
        "hold",
        "rebalance_now",
        "reduce_risk",
        "reassess_universe",
        "escalate_manual_review",
    }
    assert len(result["decision_log"]) >= 1
    assert result["freshness_map"]["monitoring_decision"] == "fresh"
    assert result["memory_refs"]


def test_critic_rejects_if_backtest_underperforms_benchmark(monkeypatch):
    from app.agents.critic import run_critic_agent
    from app.rag.schemas import DecisionMemoryQueryResult

    monkeypatch.setattr(
        "app.agents.critic.retrieve_past_mistakes",
        lambda *_args, **_kwargs: DecisionMemoryQueryResult(
            retrieval_id="memory-test",
            query="test",
            matches=[],
        ),
    )
    monkeypatch.setattr("app.agents.critic.log_decision_to_memory", lambda *_args, **_kwargs: None)

    state = {
        "profile": InvestorProfile(
            risk_profile=RiskProfile.moderate,
            horizon_years=10,
            constraints=Constraints(
                max_asset_weight=0.10,
                max_sector_weight=0.30,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.20,
                min_cash_weight=0.10,
            ),
        ),
        "market_regime": RegimeReport(
            current_regime=RegimeType.risk_off,
            confidence=0.8,
            drivers=["test"],
            is_risk_off=True,
        ),
        "proposed_portfolio": CandidatePortfolio(
            selected_assets=["SPY", "TLT"],
            weights={"SPY": 0.45, "TLT": 0.45},
            cash_weight=0.10,
            rationale=["test"],
        ),
        "backtest_result": BacktestResult(
            portfolio_total_return=0.04,
            benchmark_total_return=0.11,
            equal_weight_total_return=0.05,
            portfolio_volatility=0.12,
            portfolio_max_drawdown=0.08,
            turnover=0.0,
            observations=252,
        ),
        "risk_report": RiskReport(
            portfolio_volatility=0.12,
            max_drawdown_estimate=0.08,
            avg_correlation=0.25,
            violations=[],
            warnings=[],
            fit_to_profile="acceptable",
        ),
        "effective_policy": None,
        "revision_count": 0,
        "memory_refs": [],
        "provenance": {},
        "freshness_map": {},
        "decision_log": [],
        "critic_history": [],
    }

    result = run_critic_agent(state)
    critic_report = result["critic_report"]
    assert critic_report.verdict == CriticVerdict.reduce_risk
    assert "отстает от рынка" in critic_report.issues[0]
    assert "кэша" in critic_report.recommended_action.lower()


def test_critic_requests_more_cash_if_backtest_is_negative(monkeypatch):
    from app.agents.critic import run_critic_agent
    from app.rag.schemas import DecisionMemoryQueryResult

    monkeypatch.setattr(
        "app.agents.critic.retrieve_past_mistakes",
        lambda *_args, **_kwargs: DecisionMemoryQueryResult(
            retrieval_id="memory-test",
            query="test",
            matches=[],
        ),
    )
    monkeypatch.setattr("app.agents.critic.log_decision_to_memory", lambda *_args, **_kwargs: None)

    state = {
        "profile": InvestorProfile(
            risk_profile=RiskProfile.moderate,
            horizon_years=10,
            constraints=Constraints(
                max_asset_weight=0.10,
                max_sector_weight=0.30,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.20,
                min_cash_weight=0.10,
            ),
        ),
        "market_regime": RegimeReport(
            current_regime=RegimeType.risk_off,
            confidence=0.8,
            drivers=["test"],
            is_risk_off=True,
        ),
        "proposed_portfolio": CandidatePortfolio(
            selected_assets=["SPY", "TLT"],
            weights={"SPY": 0.45, "TLT": 0.45},
            cash_weight=0.10,
            rationale=["test"],
        ),
        "backtest_result": BacktestResult(
            portfolio_total_return=-0.03,
            benchmark_total_return=0.01,
            equal_weight_total_return=-0.01,
            portfolio_volatility=0.12,
            portfolio_max_drawdown=0.08,
            turnover=0.0,
            observations=252,
        ),
        "risk_report": RiskReport(
            portfolio_volatility=0.12,
            max_drawdown_estimate=0.08,
            avg_correlation=0.25,
            violations=[],
            warnings=[],
            fit_to_profile="acceptable",
        ),
        "effective_policy": None,
        "revision_count": 1,
        "memory_refs": [],
        "provenance": {},
        "freshness_map": {},
        "decision_log": [],
        "critic_history": [],
    }

    result = run_critic_agent(state)
    critic_report = result["critic_report"]
    assert critic_report.verdict == CriticVerdict.reduce_risk
    assert "кэша" in critic_report.recommended_action


def test_critic_accepts_small_gap_when_return_clears_hurdle(monkeypatch):
    from app.agents.critic import run_critic_agent

    monkeypatch.setattr(
        "app.agents.critic.retrieve_past_mistakes",
        lambda *_args, **_kwargs: DecisionMemoryQueryResult(
            retrieval_id="memory-test",
            query="test",
            matches=[],
        ),
    )
    monkeypatch.setattr("app.agents.critic.log_decision_to_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "app.agents.critic.get_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called for this deterministic case")),
    )

    state = {
        "profile": InvestorProfile(
            risk_profile=RiskProfile.moderate,
            horizon_years=10,
            constraints=Constraints(
                max_asset_weight=0.10,
                max_sector_weight=0.30,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.20,
                min_cash_weight=0.10,
            ),
        ),
        "market_regime": RegimeReport(
            current_regime=RegimeType.sideways,
            confidence=0.8,
            drivers=["test"],
            is_risk_off=False,
        ),
        "proposed_portfolio": CandidatePortfolio(
            selected_assets=["SPY", "TLT"],
            weights={"SPY": 0.45, "TLT": 0.45},
            cash_weight=0.10,
            rationale=["test"],
        ),
        "backtest_result": BacktestResult(
            portfolio_total_return=0.078,
            benchmark_total_return=0.110,
            equal_weight_total_return=0.080,
            portfolio_volatility=0.12,
            portfolio_max_drawdown=0.08,
            turnover=0.0,
            observations=252,
        ),
        "macro_data": MacroData(
            values={
                "vix": 18.0,
                "yield_10y": 4.1,
                "yield_3m": 2.0,
                "yield_spread": 2.1,
                "credit_spread": -0.01,
                "usd_strength": 0.0,
                "oil_mom_1m": 0.01,
                "spy_tlt_ratio": 0.02,
            },
            source="test",
        ),
        "risk_report": RiskReport(
            portfolio_volatility=0.12,
            max_drawdown_estimate=0.08,
            avg_correlation=0.25,
            violations=[],
            warnings=[],
            fit_to_profile="acceptable",
        ),
        "effective_policy": None,
        "revision_count": 1,
        "memory_refs": [],
        "provenance": {},
        "freshness_map": {},
        "decision_log": [],
        "critic_history": [],
    }

    result = run_critic_agent(state)
    assert result["critic_report"].verdict == CriticVerdict.approve
