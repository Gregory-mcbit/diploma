from app.api.routes import health_handler, run_monitoring_handler, run_portfolio_handler
from app.api.schemas import MonitoringRunRequest, PortfolioRunRequest
from app.domain.schemas import (
    BacktestResult,
    CandidatePortfolio,
    Constraints,
    CriticReport,
    CriticVerdict,
    DecisionLogEntry,
    EffectivePolicy,
    FinalRecommendation,
    FreshnessStatus,
    InvestorProfile,
    MonitoringAction,
    MonitoringDecision,
    ProvenanceRecord,
    RegimeReport,
    RegimeType,
    RiskProfile,
    RiskReport,
)


def test_health_handler():
    response = health_handler()
    assert response.status == "ok"
    assert response.service == "investment_multiagent_system"


def test_run_portfolio_handler(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.run_investment_graph",
        lambda user_query, request_id=None, correlation_id=None: {
            "request_id": request_id or "req-api-unit",
            "correlation_id": correlation_id or "corr-api-unit",
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
                inclusion_reasons={"SPY": "Core risk asset."},
                exclusion_reasons={"QQQ": "Excluded for concentration."},
                uncertainty_notes=["Signal dispersion moderate."],
                policy_summary=["Risk-off cap active."],
                memory_comparison="Compared against memory.",
            ),
            "critic_report": CriticReport(
                verdict=CriticVerdict.approve,
                issues=[],
                recommended_action="Approved.",
            ),
            "market_regime": RegimeReport(
                current_regime=RegimeType.risk_off,
                confidence=0.82,
                drivers=["Elevated volatility", "Curve inversion", "Defensive rotation"],
                is_risk_off=True,
                signal_components={"vix": 28.0},
                uncertainty_notes=["Credit signal near neutral."],
            ),
            "backtest_result": BacktestResult(
                portfolio_total_return=0.11,
                benchmark_total_return=0.09,
                equal_weight_total_return=0.08,
                portfolio_volatility=0.12,
                portfolio_max_drawdown=0.07,
                turnover=0.10,
                observations=252,
            ),
            "risk_report": RiskReport(
                portfolio_volatility=0.12,
                max_drawdown_estimate=0.09,
                avg_correlation=0.32,
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
                applied_rule_ids=["risk-off-defensive-cap"],
                applied_rule_summaries=["Risk-off regime tightened the single-name cap."],
            ),
            "decision_log": [
                DecisionLogEntry(
                    stage="critic",
                    event="portfolio_reviewed",
                    message="Approved portfolio.",
                    timestamp="2026-04-01T00:00:00Z",
                )
            ],
            "provenance": {
                "risk_report": ProvenanceRecord(
                    source="deterministic_risk_engine",
                    timestamp="2026-04-01T00:00:00Z",
                    staleness_status=FreshnessStatus.fresh,
                    confidence=0.9,
                    retrieval_id="prov-1",
                )
            },
            "freshness_map": {"risk_report": "fresh"},
            "memory_refs": [],
            "trace_log": [],
        },
    )

    response = run_portfolio_handler(PortfolioRunRequest(user_query="Build a moderate portfolio."), request_id="req-api-unit")

    assert response.final_report.portfolio.selected_assets == ["SPY", "TLT"]
    assert response.critic_report.verdict == CriticVerdict.approve
    assert response.audit.freshness_map["risk_report"] == "fresh"
    assert response.audit.request_id == "req-api-unit"
    assert response.audit.correlation_id
    assert response.backtest_result.portfolio_total_return == 0.11


def test_run_monitoring_handler(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.build_initial_state",
        lambda user_query, request_id=None, correlation_id=None: {
            "user_query": user_query,
            "request_id": request_id or "req-monitor-unit",
            "correlation_id": correlation_id or "corr-monitor-unit",
        },
    )
    monkeypatch.setattr(
        "app.api.routes.run_monitoring_graph",
        lambda initial_state: {
            "request_id": initial_state["request_id"],
            "correlation_id": initial_state["correlation_id"],
            "monitoring_decision": MonitoringDecision(
                action=MonitoringAction.rebalance_now,
                reasons=["Signal decay detected."],
                trigger_flags=["signal_decay"],
                summary="Monitoring decision: rebalance_now.",
            ),
            "market_regime": RegimeReport(
                current_regime=RegimeType.sideways,
                confidence=0.7,
                drivers=["Mixed macro", "Flat ratio", "Neutral curve"],
                is_risk_off=False,
                signal_components={"vix": 19.0},
                uncertainty_notes=["Momentum is weak."],
            ),
            "risk_report": RiskReport(
                portfolio_volatility=0.11,
                max_drawdown_estimate=0.08,
                avg_correlation=0.30,
                violations=[],
                warnings=["Rebalance threshold reached."],
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
                applied_rule_ids=["global-sector-cap"],
                applied_rule_summaries=["Global sector cap applied."],
            ),
            "decision_log": [],
            "provenance": {},
            "freshness_map": {"market_regime": "fresh"},
            "memory_refs": [],
            "trace_log": [],
        },
    )

    response = run_monitoring_handler(
        MonitoringRunRequest(
            profile=InvestorProfile(
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
            active_portfolio=CandidatePortfolio(
                selected_assets=["SPY", "TLT"],
                weights={"SPY": 0.6, "TLT": 0.3},
                cash_weight=0.1,
                rationale=["Existing portfolio."],
            ),
        ),
        request_id="req-monitor-unit",
    )

    assert response.monitoring_decision.action == MonitoringAction.rebalance_now
    assert response.audit.freshness_map["market_regime"] == "fresh"
    assert response.audit.request_id == "req-monitor-unit"
    assert response.audit.correlation_id


def test_run_portfolio_handler_terminal_rejection(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.run_investment_graph",
        lambda user_query, request_id=None, correlation_id=None: {
            "request_id": request_id or "req-api-terminal",
            "correlation_id": correlation_id or "corr-api-terminal",
            "critic_report": CriticReport(
                verdict=CriticVerdict.insufficient_confidence,
                issues=["Signals are contradictory.", "Backtest edge is weak."],
                recommended_action="Replace assets and rescore the universe.",
            ),
            "market_regime": RegimeReport(
                current_regime=RegimeType.sideways,
                confidence=0.61,
                drivers=["Mixed macro", "Flat breadth", "Weak momentum"],
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

    try:
        run_portfolio_handler(
            PortfolioRunRequest(user_query="Build a portfolio."),
            request_id="req-api-terminal",
        )
        assert False, "Expected terminal rejection RuntimeError."
    except RuntimeError as exc:
        message = str(exc)
        assert "Итоговый вердикт критика: insufficient_confidence" in message
        assert "Replace assets and rescore the universe." in message
