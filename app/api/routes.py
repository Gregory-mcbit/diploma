from __future__ import annotations

from typing import Any

try:
    from fastapi import HTTPException, Request, Response
except ImportError:
    HTTPException = Any
    Request = Any
    Response = Any

from app.api.schemas import (
    AuditPayload,
    HealthResponse,
    MonitoringRunRequest,
    MonitoringRunResponse,
    PortfolioRunRequest,
    PortfolioRunResponse,
)
from app.graph import build_initial_state, run_investment_graph, run_monitoring_graph
from app.observability.logger import get_logger
from app.observability.tracing import (
    generate_correlation_id,
    generate_request_id,
    request_trace_context,
    to_serializable,
)


logger = get_logger(__name__)


def _build_terminal_portfolio_error(state: dict) -> str:
    critic_report = state.get("critic_report")
    revision_count = state.get("revision_count", 0)
    if critic_report:
        issues = "; ".join(critic_report.issues) if critic_report.issues else "Явные проблемы не были перечислены."
        return (
            "Инвестиционный граф завершился без финальной рекомендации. "
            f"Итоговый вердикт критика: {critic_report.verdict.value}. "
            f"Рекомендованное действие: {critic_report.recommended_action}. "
            f"Проблемы: {issues}. "
            f"Количество попыток пересборки: {revision_count}."
        )
    return (
        "Инвестиционный граф завершился без финальной рекомендации и без пригодного critic_report. "
        f"Количество попыток пересборки: {revision_count}."
    )


def _build_audit_payload(state: dict) -> AuditPayload:
    return AuditPayload(
        request_id=state["request_id"],
        correlation_id=state["correlation_id"],
        decision_log=state.get("decision_log", []),
        provenance=state.get("provenance", {}),
        freshness_map=state.get("freshness_map", {}),
        memory_refs=to_serializable(state.get("memory_refs", [])),
        trace_log=state.get("trace_log", []),
    )


def health_handler() -> HealthResponse:
    return HealthResponse(status="ok", service="investment_multiagent_system")


def run_portfolio_handler(
    request: PortfolioRunRequest,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> PortfolioRunResponse:
    logger.info("API portfolio run requested.")
    resolved_request_id = request_id or generate_request_id()
    resolved_correlation_id = correlation_id or generate_correlation_id()
    with request_trace_context(resolved_request_id, resolved_correlation_id):
        result = run_investment_graph(
            request.user_query,
            request_id=resolved_request_id,
            correlation_id=resolved_correlation_id,
        )

    final_report = result.get("final_report")
    critic_report = result.get("critic_report")
    market_regime = result.get("market_regime")
    backtest_result = result.get("backtest_result")
    risk_report = result.get("risk_report")
    effective_policy = result.get("effective_policy")
    if not final_report or not critic_report or not market_regime or not backtest_result or not risk_report or not effective_policy:
        raise RuntimeError(_build_terminal_portfolio_error(result))

    return PortfolioRunResponse(
        final_report=final_report,
        critic_report=critic_report,
        market_regime=market_regime,
        backtest_result=backtest_result,
        risk_report=risk_report,
        effective_policy=effective_policy,
        profile=result.get("profile"),
        asset_scores=result.get("asset_scores", {}),
        fundamentals=result.get("fundamentals", {}),
        features=result.get("features", {}),
        news_digest=result.get("news_digest"),
        audit=_build_audit_payload(result),
    )


def run_monitoring_handler(
    request: MonitoringRunRequest,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> MonitoringRunResponse:
    logger.info("API monitoring run requested.")
    resolved_request_id = request_id or generate_request_id()
    resolved_correlation_id = correlation_id or generate_correlation_id()
    initial_state = build_initial_state(
        request.user_query,
        request_id=resolved_request_id,
        correlation_id=resolved_correlation_id,
    )
    initial_state["profile"] = request.profile
    initial_state["active_portfolio"] = request.active_portfolio
    with request_trace_context(resolved_request_id, resolved_correlation_id):
        result = run_monitoring_graph(initial_state)

    monitoring_decision = result.get("monitoring_decision")
    market_regime = result.get("market_regime")
    risk_report = result.get("risk_report")
    effective_policy = result.get("effective_policy")
    if not monitoring_decision or not market_regime or not risk_report or not effective_policy:
        raise RuntimeError("Граф мониторинга завершился без полного monitoring response payload.")

    return MonitoringRunResponse(
        monitoring_decision=monitoring_decision,
        market_regime=market_regime,
        risk_report=risk_report,
        effective_policy=effective_policy,
        audit=_build_audit_payload(result),
    )


def create_fastapi_app():
    try:
        from fastapi import FastAPI
    except ImportError as e:
        raise RuntimeError("Для создания HTTP API приложения требуется FastAPI.") from e

    app = FastAPI(title="Investment Multi-Agent System", version="1.0.0")

    @app.get("/health", response_model=HealthResponse)
    def health(http_request: Request, response: Response):
        request_id = http_request.headers.get("x-request-id") or generate_request_id()
        correlation_id = http_request.headers.get("x-correlation-id") or generate_correlation_id()
        response.headers["x-request-id"] = request_id
        response.headers["x-correlation-id"] = correlation_id
        with request_trace_context(request_id, correlation_id):
            return health_handler()

    @app.post("/portfolio/run", response_model=PortfolioRunResponse)
    def run_portfolio(request: PortfolioRunRequest, http_request: Request, response: Response):
        request_id = http_request.headers.get("x-request-id") or generate_request_id()
        correlation_id = http_request.headers.get("x-correlation-id") or generate_correlation_id()
        response.headers["x-request-id"] = request_id
        response.headers["x-correlation-id"] = correlation_id
        try:
            return run_portfolio_handler(request, request_id=request_id, correlation_id=correlation_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/monitoring/run", response_model=MonitoringRunResponse)
    def run_monitoring(request: MonitoringRunRequest, http_request: Request, response: Response):
        request_id = http_request.headers.get("x-request-id") or generate_request_id()
        correlation_id = http_request.headers.get("x-correlation-id") or generate_correlation_id()
        response.headers["x-request-id"] = request_id
        response.headers["x-correlation-id"] = correlation_id
        return run_monitoring_handler(request, request_id=request_id, correlation_id=correlation_id)

    return app
