import pytest


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.domain.schemas import (
    BacktestResult,
    CandidatePortfolio,
    Constraints,
    CriticReport,
    CriticVerdict,
    EffectivePolicy,
    FinalRecommendation,
    InvestorProfile,
    MonitoringAction,
    MonitoringDecision,
    RegimeReport,
    RegimeType,
    RiskProfile,
    RiskReport,
)


def test_http_health_endpoint():
    from app.api.app import app

    client = TestClient(app)
    response = client.get("/health", headers={"x-request-id": "req-http-health", "x-correlation-id": "corr-http-health"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-correlation-id"] == "corr-http-health"


def test_http_portfolio_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.run_investment_graph",
        lambda user_query, request_id=None, correlation_id=None: {
            "request_id": request_id or "req-http-portfolio",
            "correlation_id": correlation_id or "corr-http-portfolio",
            "final_report": FinalRecommendation(
                portfolio=CandidatePortfolio(
                    selected_assets=["SPY", "TLT"],
                    weights={"SPY": 0.6, "TLT": 0.3},
                    cash_weight=0.1,
                    rationale=["Balanced growth."],
                ),
                executive_summary="Summary",
                regime_context="Context",
                risk_disclaimer="Disclaimer",
            ),
            "critic_report": CriticReport(
                verdict=CriticVerdict.approve,
                issues=[],
                recommended_action="Approved.",
            ),
            "market_regime": RegimeReport(
                current_regime=RegimeType.risk_off,
                confidence=0.8,
                drivers=["Volatility", "Curve", "Rotation"],
                is_risk_off=True,
            ),
            "backtest_result": BacktestResult(
                portfolio_total_return=0.11,
                benchmark_total_return=0.09,
                equal_weight_total_return=0.08,
                portfolio_volatility=0.12,
                portfolio_max_drawdown=0.07,
                turnover=0.1,
                observations=252,
            ),
            "risk_report": RiskReport(
                portfolio_volatility=0.12,
                max_drawdown_estimate=0.08,
                avg_correlation=0.30,
                violations=[],
                warnings=[],
                fit_to_profile="acceptable",
            ),
            "effective_policy": EffectivePolicy(
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
            "decision_log": [],
            "provenance": {},
            "freshness_map": {},
            "memory_refs": [],
            "trace_log": [],
        },
    )

    from app.api.app import app

    client = TestClient(app)
    response = client.post(
        "/portfolio/run",
        json={"user_query": "Build a portfolio"},
        headers={"x-request-id": "req-http-portfolio", "x-correlation-id": "corr-http-portfolio"},
    )

    assert response.status_code == 200
    assert response.json()["final_report"]["portfolio"]["selected_assets"] == ["SPY", "TLT"]
    assert response.json()["audit"]["request_id"] == "req-http-portfolio"
    assert response.json()["audit"]["correlation_id"] == "corr-http-portfolio"
    assert response.headers["x-request-id"] == "req-http-portfolio"
    assert response.headers["x-correlation-id"] == "corr-http-portfolio"
    assert response.json()["backtest_result"]["portfolio_total_return"] == 0.11


def test_http_monitoring_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.build_initial_state",
        lambda user_query, request_id=None, correlation_id=None: {
            "user_query": user_query,
            "request_id": request_id or "req-http-monitor",
            "correlation_id": correlation_id or "corr-http-monitor",
        },
    )
    monkeypatch.setattr(
        "app.api.routes.run_monitoring_graph",
        lambda initial_state: {
            "request_id": initial_state["request_id"],
            "correlation_id": initial_state["correlation_id"],
            "monitoring_decision": MonitoringDecision(
                action=MonitoringAction.rebalance_now,
                reasons=["Signal decay"],
                trigger_flags=["signal_decay"],
                summary="Monitoring decision: rebalance_now.",
            ),
            "market_regime": RegimeReport(
                current_regime=RegimeType.sideways,
                confidence=0.7,
                drivers=["Mixed macro", "Flat ratio", "Neutral curve"],
                is_risk_off=False,
            ),
            "risk_report": RiskReport(
                portfolio_volatility=0.11,
                max_drawdown_estimate=0.08,
                avg_correlation=0.30,
                violations=[],
                warnings=[],
                fit_to_profile="acceptable",
            ),
            "effective_policy": EffectivePolicy(
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
            "decision_log": [],
            "provenance": {},
            "freshness_map": {},
            "memory_refs": [],
            "trace_log": [],
        },
    )

    from app.api.app import app

    client = TestClient(app)
    response = client.post(
        "/monitoring/run",
        json={
            "profile": {
                "risk_profile": "moderate",
                "horizon_years": 10,
                "target": "Balanced growth",
                "constraints": {
                    "max_asset_weight": 0.25,
                    "max_sector_weight": 0.35,
                    "allowed_asset_classes": ["stocks", "bonds", "commodities"],
                    "forbidden_assets": [],
                    "max_drawdown_tolerance": 0.20,
                    "min_cash_weight": 0.02,
                    "max_correlation_threshold": 0.85,
                },
                "rebalancing_policy": {
                    "mode": "threshold_and_periodic",
                    "period_days": 30,
                    "drift_threshold": 0.05,
                    "review_frequency": "monthly",
                },
            },
            "active_portfolio": {
                "selected_assets": ["SPY", "TLT"],
                "weights": {"SPY": 0.6, "TLT": 0.3},
                "cash_weight": 0.1,
                "rationale": ["Existing portfolio."],
            },
            "user_query": "monitoring",
        },
        headers={"x-request-id": "req-http-monitor", "x-correlation-id": "corr-http-monitor"},
    )

    assert response.status_code == 200
    assert response.json()["monitoring_decision"]["action"] == "rebalance_now"
    assert response.json()["audit"]["request_id"] == "req-http-monitor"
    assert response.json()["audit"]["correlation_id"] == "corr-http-monitor"
    assert response.headers["x-request-id"] == "req-http-monitor"
    assert response.headers["x-correlation-id"] == "corr-http-monitor"


def test_http_portfolio_endpoint_terminal_rejection(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.run_investment_graph",
        lambda user_query, request_id=None, correlation_id=None: {
            "request_id": request_id or "req-http-terminal",
            "correlation_id": correlation_id or "corr-http-terminal",
            "critic_report": CriticReport(
                verdict=CriticVerdict.insufficient_confidence,
                issues=["Signals are contradictory.", "Backtest edge is weak."],
                recommended_action="Replace assets and rescore the universe.",
            ),
            "market_regime": RegimeReport(
                current_regime=RegimeType.sideways,
                confidence=0.62,
                drivers=["Mixed macro", "Flat ratio", "Weak breadth"],
                is_risk_off=False,
            ),
            "backtest_result": BacktestResult(
                portfolio_total_return=0.02,
                benchmark_total_return=0.03,
                equal_weight_total_return=0.025,
                portfolio_volatility=0.10,
                portfolio_max_drawdown=0.06,
                turnover=0.08,
                observations=252,
            ),
            "risk_report": RiskReport(
                portfolio_volatility=0.10,
                max_drawdown_estimate=0.07,
                avg_correlation=0.31,
                violations=[],
                warnings=["Conviction spread is narrow."],
                fit_to_profile="acceptable",
            ),
            "effective_policy": EffectivePolicy(
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
            "revision_count": 2,
            "decision_log": [],
            "provenance": {},
            "freshness_map": {},
            "memory_refs": [],
            "trace_log": [],
        },
    )

    from app.api.app import app

    client = TestClient(app)
    response = client.post(
        "/portfolio/run",
        json={"user_query": "Build a portfolio"},
        headers={"x-request-id": "req-http-terminal", "x-correlation-id": "corr-http-terminal"},
    )

    assert response.status_code == 409
    assert "Итоговый вердикт критика: insufficient_confidence" in response.json()["detail"]
