from __future__ import annotations

from typing import Literal
import pandas as pd

from langgraph.graph import END, START, StateGraph

from app.agents.critic import run_critic_agent
from app.agents.data import run_data_agent
from app.agents.explainability import run_explainability_agent
from app.agents.monitoring import run_monitoring_agent
from app.agents.portfolio import run_portfolio_agent
from app.agents.profile import run_profile_agent
from app.agents.regime import run_regime_agent
from app.agents.risk import run_risk_agent
from app.agents.scoring import run_scoring_agent
from app.domain.schemas import CriticVerdict, FreshnessStatus
from app.observability.logger import get_logger
from app.observability.trace_store import persist_execution_trace
from app.observability.tracing import (
    generate_correlation_id,
    generate_request_id,
    request_trace_context,
)
from app.graph.state import GraphState
from app.graph.telemetry import append_decision_log, append_trace_event, build_provenance_record
from app.policy_engine import derive_effective_policy
from app.tools.backtest import backtest_portfolio
from app.domain.schemas import ExecutionTraceRecord
from app.graph.telemetry import utc_now_iso

MAX_REVISIONS = 3
logger = get_logger(__name__)


def _route_after_critic(
    state: GraphState,
) -> Literal["approved", "revise_weights", "replace_assets", "reduce_risk", "insufficient_confidence", "terminal_report"]:
    critic_report = state.get("critic_report")
    revision_count = state.get("revision_count", 0)

    if not critic_report:
        raise RuntimeError("Для маршрутизации после критика требуется critic_report в graph state.")

    if critic_report.verdict == CriticVerdict.approve:
        return "approved"

    if revision_count >= MAX_REVISIONS:
        return "terminal_report"

    if critic_report.verdict == CriticVerdict.replace_assets:
        return "replace_assets"
    if critic_report.verdict == CriticVerdict.reduce_risk:
        return "reduce_risk"
    if critic_report.verdict == CriticVerdict.insufficient_confidence:
        return "terminal_report"
    return "revise_weights"


def _increment_revision_count(state: GraphState) -> dict:
    return {"revision_count": state.get("revision_count", 0) + 1}


def _wrap_node(node_name: str, func):
    def _wrapped(state: GraphState) -> dict:
        with request_trace_context(state.get("request_id"), state.get("correlation_id")):
            trace_log = append_trace_event(
                state,
                stage=node_name,
                event="node_started",
                message=f"Нода {node_name} запущена.",
            )
            try:
                updates = func(state)
            except Exception as exc:
                failed_state = dict(state)
                failed_state["trace_log"] = trace_log
                failed_trace = append_trace_event(
                    failed_state,
                    stage=node_name,
                    event="node_failed",
                    message=f"Нода {node_name} завершилась ошибкой.",
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                )
                state["trace_log"] = failed_trace
                raise
            merged_state = dict(state)
            merged_state.update(updates)
            merged_state["trace_log"] = list(updates.get("trace_log", trace_log))
            completed_trace = append_trace_event(
                merged_state,
                stage=node_name,
                event="node_completed",
                message=f"Нода {node_name} завершена.",
            )
            updates["trace_log"] = completed_trace
            return updates

    return _wrapped


def _persist_trace_record(state: GraphState, *, run_type: str, status: str, started_at: str, error: str | None = None) -> None:
    record = ExecutionTraceRecord(
        request_id=state["request_id"],
        correlation_id=state["correlation_id"],
        run_type=run_type,
        status=status,
        started_at=started_at,
        completed_at=utc_now_iso(),
        trace_log=state.get("trace_log", []),
        decision_log=state.get("decision_log", []),
        metadata=(
            {"error": error} if error else {}
        ),
    )
    persist_execution_trace(record)


def _halt_for_insufficient_confidence(state: GraphState) -> dict:
    critic_report = state.get("critic_report")
    if not critic_report:
        raise RuntimeError("Для остановки по insufficient confidence требуется critic_report в graph state.")

    decision_log = append_decision_log(
        state,
        stage="halt",
        event="insufficient_confidence",
        message="Пайплайн остановлен без финальной рекомендации после вердикта insufficient_confidence.",
        metadata={"verdict": critic_report.verdict.value},
    )
    trace_log = append_trace_event(
        state,
        stage="halt",
        event="pipeline_halted",
        message="Пайплайн остановлен после вердикта insufficient_confidence.",
        metadata={"verdict": critic_report.verdict.value},
    )
    return {"decision_log": decision_log, "trace_log": trace_log}


def run_policy_node(state: GraphState) -> dict:
    profile = state.get("profile")
    regime = state.get("market_regime")
    if not profile or not regime:
        raise ValueError("Policy node требует profile и market_regime.")

    effective_policy = derive_effective_policy(profile, regime)
    provenance = dict(state.get("provenance", {}))
    provenance["effective_policy"] = build_provenance_record(
        source="policy_engine",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.98,
        details={"rule_count": len(effective_policy.applied_rule_ids)},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["effective_policy"] = FreshnessStatus.fresh.value
    decision_log = append_decision_log(
        state,
        stage="policy",
        event="effective_policy_derived",
        message=f"Сформирована effective policy на основе {len(effective_policy.applied_rule_ids)} структурированных правил.",
        rule_ids=effective_policy.applied_rule_ids,
        metadata={
            "max_asset_weight": effective_policy.constraints.max_asset_weight,
            "min_cash_weight": effective_policy.constraints.min_cash_weight,
        },
    )
    return {
        "effective_policy": effective_policy,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "decision_log": decision_log,
    }


def run_backtest_node(state: GraphState) -> dict:
    portfolio = state.get("proposed_portfolio")
    parquet_pointer = state.get("market_data_pointer")
    if not portfolio or not parquet_pointer:
        raise ValueError("Backtest node требует proposed_portfolio и market_data_pointer.")

    price_df = pd.read_parquet(parquet_pointer)
    previous_portfolio = state.get("active_portfolio")
    backtest_result = backtest_portfolio(
        price_df=price_df,
        weights=portfolio.weights,
        cash_weight=portfolio.cash_weight,
        benchmark_ticker="SPY",
        previous_weights=(previous_portfolio.weights if previous_portfolio else None),
    )

    provenance = dict(state.get("provenance", {}))
    provenance["backtest_result"] = build_provenance_record(
        source="backtest_portfolio",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.86,
        details={"observations": backtest_result.observations},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["backtest_result"] = FreshnessStatus.fresh.value
    decision_log = append_decision_log(
        state,
        stage="backtest",
        event="portfolio_backtested",
        message="Кандидатный портфель прогнан через бэктест против бенчмарка и equal-weight baseline.",
        tool_calls=["backtest_portfolio"],
        metadata={
            "portfolio_total_return": backtest_result.portfolio_total_return,
            "benchmark_total_return": backtest_result.benchmark_total_return,
            "turnover": backtest_result.turnover,
        },
    )
    return {
        "backtest_result": backtest_result,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "decision_log": decision_log,
    }


def _activate_current_portfolio(state: GraphState) -> dict:
    portfolio = state.get("proposed_portfolio")
    if not portfolio:
        raise RuntimeError("Activation node требует proposed_portfolio в graph state.")
    decision_log = append_decision_log(
        state,
        stage="activation",
        event="portfolio_activated",
        message="Последний proposed portfolio переведен в active portfolio state.",
    )
    return {
        "active_portfolio": portfolio,
        "provenance": {
            **dict(state.get("provenance", {})),
            "active_portfolio": build_provenance_record(
                source="activation_node",
                staleness_status=FreshnessStatus.fresh,
                confidence=1.0,
                details={"selected_assets": len(portfolio.selected_assets)},
            ),
        },
        "freshness_map": {
            **dict(state.get("freshness_map", {})),
            "active_portfolio": FreshnessStatus.fresh.value,
        },
        "decision_log": decision_log,
    }


def build_investment_graph():
    graph = StateGraph(GraphState)

    graph.add_node("profile", _wrap_node("profile", run_profile_agent))
    graph.add_node("data", _wrap_node("data", run_data_agent))
    graph.add_node("scoring", _wrap_node("scoring", run_scoring_agent))
    graph.add_node("regime", _wrap_node("regime", run_regime_agent))
    graph.add_node("policy", _wrap_node("policy", run_policy_node))
    graph.add_node("portfolio", _wrap_node("portfolio", run_portfolio_agent))
    graph.add_node("backtest", _wrap_node("backtest", run_backtest_node))
    graph.add_node("risk", _wrap_node("risk", run_risk_agent))
    graph.add_node("critic", _wrap_node("critic", run_critic_agent))
    graph.add_node("increment_revision_portfolio", _wrap_node("increment_revision_portfolio", _increment_revision_count))
    graph.add_node("increment_revision_scoring", _wrap_node("increment_revision_scoring", _increment_revision_count))
    graph.add_node("explainability", _wrap_node("explainability", run_explainability_agent))
    graph.add_node("halt_insufficient_confidence", _wrap_node("halt_insufficient_confidence", _halt_for_insufficient_confidence))
    graph.add_node("activate_portfolio", _wrap_node("activate_portfolio", _activate_current_portfolio))
    graph.add_node("monitoring", _wrap_node("monitoring", run_monitoring_agent))

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "data")
    graph.add_edge("data", "scoring")
    graph.add_edge("scoring", "regime")
    graph.add_edge("regime", "policy")
    graph.add_edge("policy", "portfolio")
    graph.add_edge("portfolio", "backtest")
    graph.add_edge("backtest", "risk")
    graph.add_edge("risk", "critic")

    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "approved": "activate_portfolio",
            "revise_weights": "increment_revision_portfolio",
            "replace_assets": "increment_revision_scoring",
            "reduce_risk": "increment_revision_portfolio",
            "insufficient_confidence": "halt_insufficient_confidence",
            "terminal_report": "explainability",
        },
    )
    graph.add_edge("increment_revision_portfolio", "portfolio")
    graph.add_edge("increment_revision_scoring", "scoring")
    graph.add_edge("activate_portfolio", "explainability")
    graph.add_edge("explainability", END)
    graph.add_edge("halt_insufficient_confidence", END)

    return graph.compile()


def build_monitoring_graph():
    graph = StateGraph(GraphState)

    graph.add_node("data", _wrap_node("data", run_data_agent))
    graph.add_node("scoring", _wrap_node("scoring", run_scoring_agent))
    graph.add_node("regime", _wrap_node("regime", run_regime_agent))
    graph.add_node("policy", _wrap_node("policy", run_policy_node))
    graph.add_node("risk", _wrap_node("risk", run_risk_agent))
    graph.add_node("monitoring", _wrap_node("monitoring", run_monitoring_agent))

    graph.add_edge(START, "data")
    graph.add_edge("data", "scoring")
    graph.add_edge("scoring", "regime")
    graph.add_edge("regime", "policy")
    graph.add_edge("policy", "risk")
    graph.add_edge("risk", "monitoring")
    graph.add_edge("monitoring", END)

    return graph.compile()


def build_initial_state(
    user_query: str,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> GraphState:
    return GraphState(
        user_query=user_query,
        request_id=request_id or generate_request_id(),
        correlation_id=correlation_id or generate_correlation_id(),
        revision_count=0,
        critic_history=[],
        provenance={},
        freshness_map={},
        memory_refs=[],
        decision_log=[],
        fundamentals={},
        macro_data=None,
        features={},
        news_articles={},
        news_digest=None,
        backtest_result=None,
        trace_log=[],
    )


def run_investment_graph(
    user_query: str,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> GraphState:
    app = build_investment_graph()
    initial_state = build_initial_state(user_query, request_id=request_id, correlation_id=correlation_id)
    started_at = utc_now_iso()
    with request_trace_context(initial_state["request_id"], initial_state["correlation_id"]):
        initial_state["trace_log"] = append_trace_event(
            initial_state,
            stage="graph",
            event="execution_started",
            message="Investment graph execution started.",
        )
        logger.info("Starting investment graph execution.")
        try:
            result = app.invoke(initial_state)
            result["trace_log"] = append_trace_event(
                result,
                stage="graph",
                event="execution_completed",
                message="Investment graph execution completed.",
            )
            _persist_trace_record(result, run_type="investment", status="completed", started_at=started_at)
            logger.info("Completed investment graph execution.")
            return result
        except Exception as exc:
            initial_state["trace_log"] = append_trace_event(
                initial_state,
                stage="graph",
                event="execution_failed",
                message="Investment graph execution failed.",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            _persist_trace_record(initial_state, run_type="investment", status="failed", started_at=started_at, error=str(exc))
            raise


def run_monitoring_graph(initial_state: GraphState) -> GraphState:
    app = build_monitoring_graph()
    request_id = initial_state.get("request_id") or generate_request_id()
    correlation_id = initial_state.get("correlation_id") or generate_correlation_id()
    initial_state["request_id"] = request_id
    initial_state["correlation_id"] = correlation_id
    if "trace_log" not in initial_state:
        initial_state["trace_log"] = []
    started_at = utc_now_iso()
    with request_trace_context(request_id, correlation_id):
        initial_state["trace_log"] = append_trace_event(
            initial_state,
            stage="graph",
            event="execution_started",
            message="Monitoring graph execution started.",
        )
        logger.info("Starting monitoring graph execution.")
        try:
            result = app.invoke(initial_state)
            result["trace_log"] = append_trace_event(
                result,
                stage="graph",
                event="execution_completed",
                message="Monitoring graph execution completed.",
            )
            _persist_trace_record(result, run_type="monitoring", status="completed", started_at=started_at)
            logger.info("Completed monitoring graph execution.")
            return result
        except Exception as exc:
            initial_state["trace_log"] = append_trace_event(
                initial_state,
                stage="graph",
                event="execution_failed",
                message="Monitoring graph execution failed.",
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            _persist_trace_record(initial_state, run_type="monitoring", status="failed", started_at=started_at, error=str(exc))
            raise
