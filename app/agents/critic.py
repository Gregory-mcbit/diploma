import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from app.agents.base import get_llm
from app.domain.schemas import CriticReport, CriticVerdict, FreshnessStatus, MemoryReference
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.rag.decision_memory_rag import log_decision_to_memory, retrieve_past_mistakes
logger = get_logger(__name__)

BENCHMARK_UNDERPERFORMANCE_TOLERANCE = 0.05


def _backtest_hurdle_proxy(state: GraphState) -> float:
    backtest_result = state["backtest_result"]
    macro_payload = state.get("macro_data")
    if not macro_payload:
        return 0.0
    macro_values = getattr(macro_payload, "values", {}) or {}
    short_rate = float(macro_values.get("yield_3m", 0.0))
    if short_rate <= 0.0:
        return 0.0
    observations = max(1, int(backtest_result.observations))
    return (short_rate / 100.0) * (observations / 252.0)


def _build_deterministic_critic_report(state: GraphState) -> CriticReport | None:
    risk_report = state["risk_report"]
    backtest_result = state["backtest_result"]

    if risk_report.violations:
        return CriticReport(
            verdict=CriticVerdict.revise_weights,
            issues=list(risk_report.violations),
            recommended_action=(
                "Портфель нарушает жесткие risk/policy ограничения. "
                "Пересобери веса так, чтобы убрать математические нарушения, прежде чем двигаться дальше."
            ),
        )

    portfolio_total_return = float(backtest_result.portfolio_total_return)
    benchmark_total_return = float(backtest_result.benchmark_total_return)
    hurdle_proxy = _backtest_hurdle_proxy(state)

    if portfolio_total_return < 0.0:
        return CriticReport(
            verdict=CriticVerdict.reduce_risk,
            issues=[
                (
                    "Бэктест портфеля остается отрицательным: "
                    f"{portfolio_total_return:.2%}."
                ),
                (
                    "При таком результате система не должна наращивать риск, "
                    "пока не появится более устойчивый набор активов."
                ),
            ],
            recommended_action=(
                "Снизь риск портфеля и подними долю кэша. "
                "Если после пересборки ожидаемая конструкция все еще уходит в отрицательную доходность, "
                "предпочти значимый кэш-буфер вместо агрессивной аллокации."
            ),
        )

    if portfolio_total_return < benchmark_total_return:
        gap = benchmark_total_return - portfolio_total_return
        if gap <= BENCHMARK_UNDERPERFORMANCE_TOLERANCE and portfolio_total_return > hurdle_proxy:
            return CriticReport(
                verdict=CriticVerdict.approve,
                issues=[
                    (
                        "Портфель слегка уступает бенчмарку, но отставание остается в допустимом мягком диапазоне "
                        f"{gap:.2%} и результат выше proxy-порога {hurdle_proxy:.2%}."
                    )
                ],
                recommended_action=(
                    "Портфель можно принять как рабочую конструкцию. В отчете нужно явно показать, "
                    "что он немного уступил рынку, но сохранил приемлемую доходность и не требует аварийного ухода в кэш."
                ),
            )
        return CriticReport(
            verdict=CriticVerdict.reduce_risk,
            issues=[
                (
                    "Портфель заметно отстает от рынка в бэктесте: "
                    f"{portfolio_total_return:.2%} против {benchmark_total_return:.2%} у бенчмарка."
                ),
                (
                    f"Разрыв составляет {gap:.2%}, что выше допустимого soft-порога "
                    f"{BENCHMARK_UNDERPERFORMANCE_TOLERANCE:.2%}."
                ),
            ],
            recommended_action=(
                "Не форсируй риск ради формальной аллокации. Подними долю кэша и сузь рискованный sleeve, "
                "если после пересборки портфель все равно заметно уступает рынку."
            ),
        )

    return None

def run_critic_agent(state: GraphState) -> Dict[str, Any]:
    """
    Critic Agent: The Autonomous Supervisor.
    1. Checks Decision Memory RAG for past mistakes.
    2. Reads RiskReport to guarantee mathematical compliance.
    3. Outputs 'approve' or rejects with precise string instructions.
    """
    logger.info("Running Critic Agent.")
    
    profile = state.get("profile")
    regime = state.get("market_regime")
    portfolio = state.get("proposed_portfolio")
    backtest_result = state.get("backtest_result")
    risk_report = state.get("risk_report")
    effective_policy = state.get("effective_policy")
    iteration = state.get("revision_count", 0)
    
    if not profile or not regime or not portfolio or not backtest_result or not risk_report:
        raise ValueError("Critic Agent requires profile, regime, proposed_portfolio, backtest_result, and risk_report.")
        
    # RAG Memory Recall
    past_mistakes_result = retrieve_past_mistakes(profile.risk_profile.value, regime.current_regime.value)
    memory_refs = list(state.get("memory_refs", []))
    memory_refs.append(
        MemoryReference(
            layer="decision_memory",
            retrieval_id=past_mistakes_result.retrieval_id,
            summary=f"Для проверки извлечено {len(past_mistakes_result.matches)} прошлых решений критика.",
            source=past_mistakes_result.source,
        )
    )
    
    deterministic_report = _build_deterministic_critic_report(state)

    if deterministic_report is None:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(CriticReport, method="function_calling")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты главный инвестиционный критик автономной агентной системы.
Твоя задача — проверить математически построенный портфель, предупреждения по риску и память RAG о ПРОШЛЫХ ОШИБКАХ.

ЖЕСТКИЕ ПРАВИЛА:
1. Если есть математические нарушения, ты ОБЯЗАН отклонить портфель, выставив `verdict` как 'revise_weights' или 'replace_assets'. Никогда не одобряй математические нарушения.
2. Если прошлые ошибки показывают, что такой набор активов уже приводил к плохому результату, отклони его и потребуй другой подход.
3. Если доходность портфеля ниже бенчмарка или отрицательна, ты НЕ МОЖЕШЬ одобрять портфель.
4. Если нарушений НЕТ, и портфель выглядит фундаментально вменяемым, ты ОБЯЗАН выставить `verdict` = 'approve'.
4. Всегда давай конкретное `recommended_action`, чтобы Portfolio Agent понимал, какие веса или активы нужно изменить на повторной итерации.

Контекст прошлых ошибок критика (JSON-массив прежних неудач):
{past_mistakes}
"""),
            ("human", "Профиль: {profile}\nРежим: {regime}\nВеса портфеля: {weights}\nСводка бэктеста: {backtest_summary}\nМатематические нарушения: {violations}\nКоличество попыток: {iteration}\n\nСформируй CriticReport на русском языке.")
        ])
        
        chain = prompt | structured_llm
        
        critic_report: CriticReport = chain.invoke({
            "past_mistakes": json.dumps(
                [match.model_dump() for match in past_mistakes_result.matches],
                ensure_ascii=True,
            ),
            "profile": profile.risk_profile.value,
            "regime": regime.current_regime.value,
            "weights": json.dumps(portfolio.weights),
            "violations": risk_report.violations,
            "iteration": iteration,
            "backtest_summary": json.dumps(
                {
                    "portfolio_total_return": backtest_result.portfolio_total_return,
                    "benchmark_total_return": backtest_result.benchmark_total_return,
                    "equal_weight_total_return": backtest_result.equal_weight_total_return,
                    "turnover": backtest_result.turnover,
                    "max_drawdown": backtest_result.portfolio_max_drawdown,
                }
            ),
        })
    else:
        critic_report = deterministic_report
    
    logger.info("Critic verdict=%s.", critic_report.verdict.value)

    critic_history = list(state.get("critic_history", []))
    critic_history.append(critic_report)
    
    # RAG Memory Insertion: If Critic rejected it, log this failure so it remembers next loop!
    if critic_report.verdict.value != "approve":
        log_decision_to_memory(
            profile_type=profile.risk_profile.value,
            regime=regime.current_regime.value,
            portfolio=portfolio,
            critic_report=critic_report
        )
    provenance = dict(state.get("provenance", {}))
    provenance["critic_report"] = build_provenance_record(
        source=("critic_deterministic_rules" if deterministic_report is not None else "critic_llm_with_memory"),
        staleness_status=FreshnessStatus.fresh,
        confidence=0.75,
        details={"verdict": critic_report.verdict.value},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["decision_memory"] = FreshnessStatus.fresh.value
    freshness_map["critic_report"] = FreshnessStatus.fresh.value
    provenance["decision_memory_critic"] = build_provenance_record(
        source=past_mistakes_result.source,
        staleness_status=FreshnessStatus.fresh,
        confidence=0.80,
        retrieval_id=past_mistakes_result.retrieval_id,
        details={"match_count": len(past_mistakes_result.matches)},
    )
    decision_log = append_decision_log(
        state,
        stage="critic",
        event="portfolio_reviewed",
        message=f"Критик вернул вердикт {critic_report.verdict.value}.",
        tool_calls=["retrieve_past_mistakes", "log_decision_to_memory"],
        rule_ids=(effective_policy.applied_rule_ids if effective_policy else []),
        metadata={
            "verdict": critic_report.verdict.value,
            "issue_count": len(critic_report.issues),
        },
    )
    return {
        "critic_report": critic_report,
        "critic_history": critic_history,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "memory_refs": memory_refs,
        "decision_log": decision_log,
    }
