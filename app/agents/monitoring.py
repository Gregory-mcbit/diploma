from __future__ import annotations

from typing import Dict, Any

from app.domain.schemas import AssetScore, FreshnessStatus, MemoryReference, MonitoringAction, MonitoringDecision
from app.observability.logger import get_logger
from app.graph.state import GraphState
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.rag.decision_memory_rag import log_monitoring_decision, retrieve_past_mistakes
from app.tools.optimizer import _coerce_score
logger = get_logger(__name__)


def run_monitoring_agent(state: GraphState) -> Dict[str, Any]:
    """
    Monitoring Agent: evaluates an active portfolio against refreshed scores,
    regime, and risk to decide whether to hold, rebalance, reduce risk, or
    escalate for manual review.
    """
    logger.info("Running Monitoring Agent.")

    profile = state.get("profile")
    active_portfolio = state.get("active_portfolio") or state.get("proposed_portfolio")
    asset_scores = state.get("asset_scores", {})
    regime = state.get("market_regime")
    risk_report = state.get("risk_report")
    effective_policy = state.get("effective_policy")

    if not profile or not active_portfolio or not regime or not risk_report:
        raise ValueError("Monitoring Agent требует profile, active/proposed portfolio, regime и risk_report.")
    if not asset_scores:
        raise ValueError("Monitoring Agent требует asset_scores для проверки дрейфа и сигналов.")

    trigger_flags: list[str] = []
    reasons: list[str] = []

    if risk_report.violations:
        trigger_flags.append("risk_threshold_breach")
        reasons.append("Валидация риска обнаружила жесткие нарушения policy.")

    selected_scores = {
        ticker: _coerce_score(asset_scores.get(ticker, 0.0))
        for ticker in active_portfolio.selected_assets
    }
    if any(score < 0 for score in selected_scores.values()):
        trigger_flags.append("signal_decay")
        reasons.append("Как минимум одна активная позиция теперь имеет отрицательный модельный скор.")

    if regime.is_risk_off:
        trigger_flags.append("regime_shift")
        reasons.append("Текущий режим является risk-off и требует более пристального контроля.")

    if effective_policy:
        max_weight = max(active_portfolio.weights.values(), default=0.0)
        if max_weight > effective_policy.constraints.max_asset_weight + 0.001:
            trigger_flags.append("weight_drift")
            reasons.append("Наблюдаемые веса превышают активный policy cap.")

    memory_result = retrieve_past_mistakes(
        profile.risk_profile.value,
        regime.current_regime.value,
    )
    memory_refs = list(state.get("memory_refs", []))
    memory_refs.append(
        MemoryReference(
            layer="decision_memory",
            retrieval_id=memory_result.retrieval_id,
            summary=f"Для мониторинга извлечено {len(memory_result.matches)} прошлых monitoring/critic исходов.",
            source=memory_result.source,
        )
    )
    if memory_result.matches:
        trigger_flags.append("memory_caution")
        reasons.append("В decision memory есть связанные прошлые неудачи в похожих условиях.")

    if "risk_threshold_breach" in trigger_flags:
        action = MonitoringAction.reduce_risk
    elif "weight_drift" in trigger_flags or "signal_decay" in trigger_flags:
        action = MonitoringAction.rebalance_now
    elif "memory_caution" in trigger_flags:
        action = MonitoringAction.escalate_manual_review
    elif "regime_shift" in trigger_flags:
        action = MonitoringAction.reassess_universe
    else:
        action = MonitoringAction.hold
        reasons.append("Материального дрейфа, деградации сигналов или нарушения policy не обнаружено.")

    decision = MonitoringDecision(
        action=action,
        reasons=reasons,
        trigger_flags=trigger_flags,
        summary=f"Решение мониторинга: {action.value}.",
    )

    log_monitoring_decision(
        profile_type=profile.risk_profile.value,
        regime=regime.current_regime.value,
        portfolio=active_portfolio,
        monitoring_decision=decision,
    )

    provenance = dict(state.get("provenance", {}))
    provenance["monitoring_decision"] = build_provenance_record(
        source="deterministic_monitoring_agent",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.82,
        details={"action": decision.action.value},
    )
    provenance["decision_memory_monitoring"] = build_provenance_record(
        source=memory_result.source,
        staleness_status=FreshnessStatus.fresh,
        confidence=0.80,
        retrieval_id=memory_result.retrieval_id,
        details={"match_count": len(memory_result.matches)},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["decision_memory"] = FreshnessStatus.fresh.value
    freshness_map["monitoring_decision"] = FreshnessStatus.fresh.value
    decision_log = append_decision_log(
        state,
        stage="monitoring",
        event="monitoring_decision_made",
        message=f"Monitoring agent рекомендовал действие {decision.action.value}.",
        tool_calls=["retrieve_past_mistakes", "log_monitoring_decision"],
        rule_ids=(effective_policy.applied_rule_ids if effective_policy else []),
        metadata={"trigger_flags": trigger_flags},
    )
    return {
        "monitoring_decision": decision,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "memory_refs": memory_refs,
        "decision_log": decision_log,
    }
