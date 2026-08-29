import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.agents.base import get_llm
from app.domain.schemas import FinalRecommendation, FreshnessStatus, MemoryReference
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.rag.knowledge_rag import retrieve_rules
from app.rag.decision_memory_rag import retrieve_past_mistakes
logger = get_logger(__name__)


class ExplainabilitySections(BaseModel):
    executive_summary: str
    regime_context: str
    risk_disclaimer: str


class ExplainabilityInputPayload(BaseModel):
    critic_verdict: str
    critic_action: str
    revision_count: int = 0
    regime: str
    policy_summaries: list[str] = Field(default_factory=list)
    methodology_matches: list[dict] = Field(default_factory=list)
    construction_notes: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    memory_matches: list[dict] = Field(default_factory=list)
    selected_assets: list[str] = Field(default_factory=list)
    weights: Dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    score_summary: list[dict] = Field(default_factory=list)
    backtest_summary: list[str] = Field(default_factory=list)
    process_summary: list[str] = Field(default_factory=list)


def _build_backtest_summary(state: GraphState) -> list[str]:
    backtest_result = state.get("backtest_result")
    critic_report = state.get("critic_report")
    macro_payload = state.get("macro_data")
    if not backtest_result:
        return []

    summary = [
        "Бэктест проверил фиксированные веса последней конструкции портфеля на исторических дневных ценах выбранных активов.",
        "Результат сравнивался с бенчмарком SPY и с equal-weight корзиной из тех же активов, а turnover считался против предыдущей структуры портфеля, если она была доступна.",
        (
            f"Итог на текущем окне: портфель {backtest_result.portfolio_total_return:.2%}, "
            f"рынок {backtest_result.benchmark_total_return:.2%}, equal-weight {backtest_result.equal_weight_total_return:.2%}, "
            f"геометрическая доходность {backtest_result.portfolio_geometric_mean_return:.2%}, "
            f"волатильность {backtest_result.portfolio_volatility:.2%}, максимальная просадка {backtest_result.portfolio_max_drawdown:.2%}, "
            f"наблюдений {backtest_result.observations}."
        ),
    ]
    if macro_payload and getattr(macro_payload, "values", None):
        yield_3m = float(macro_payload.values.get("yield_3m", 0.0))
        hurdle = (yield_3m / 100.0) * (max(1, int(backtest_result.observations)) / 252.0)
        if yield_3m > 0.0:
            summary.append(
                f"Для мягкой оценки реальной отдачи использован текущий proxy-порог на основе 3M Treasury: около {hurdle:.2%} за сопоставимый период."
            )
    if critic_report and critic_report.verdict.value != "approve":
        summary.append(
            f"Итог проверки критика: {critic_report.verdict.value}. Это значит, что система решила не форсировать полное одобрение текущей risk-аллокации."
        )
    return summary


def _build_process_summary(state: GraphState) -> list[str]:
    decision_log = state.get("decision_log", [])
    summaries: list[str] = []
    attempt_no = 0
    for entry in decision_log:
        if entry.stage != "portfolio" or entry.event != "portfolio_constructed":
            continue
        attempt_no += 1
        metadata = entry.metadata or {}
        selected_assets = metadata.get("selected_assets", [])
        if not isinstance(selected_assets, list):
            selected_assets = []
        allocation_mode = metadata.get("allocation_mode", "n/a")
        cash_weight = metadata.get("cash_weight")
        cash_text = ""
        if isinstance(cash_weight, (float, int)):
            cash_text = f", кэш {float(cash_weight):.1%}"
        asset_text = ", ".join(selected_assets[:8]) if selected_assets else "без явного списка активов"
        summaries.append(
            f"Попытка {attempt_no}: тестировался набор {asset_text}; режим аллокации {allocation_mode}{cash_text}."
        )
    critic_report = state.get("critic_report")
    revision_count = int(state.get("revision_count", 0))
    if critic_report and critic_report.verdict.value != "approve":
        summaries.append(
            f"После {revision_count} пересборок система не получила полностью устраивающий критика результат и перешла к объяснению лучшей найденной конструкции с более защитительной трактовкой."
        )
    return summaries


def run_explainability_agent(state: GraphState) -> Dict[str, Any]:
    """
    Explainability Agent: Translates the final approved quantitative portfolio
    into a beautiful human-readable report.
    LLM writes ONLY the three text sections; FinalRecommendation is assembled in Python.
    """
    logger.info("Running Explainability Agent.")

    portfolio = state.get("proposed_portfolio")
    regime = state.get("market_regime")
    critic_report = state.get("critic_report")
    profile = state.get("profile")
    effective_policy = state.get("effective_policy")

    if not regime or not portfolio or not critic_report or not profile:
        raise ValueError("Explainability Agent requires profile, portfolio, regime, and critic_report.")

    memory_result = retrieve_past_mistakes(profile.risk_profile.value, regime.current_regime.value)
    memory_comparison = None
    memory_refs = list(state.get("memory_refs", []))
    memory_refs.append(
        MemoryReference(
            layer="decision_memory",
            retrieval_id=memory_result.retrieval_id,
            summary=(
                "Сопоставили текущую конструкцию портфеля с ранее отклоненными шаблонами в decision memory. "
                if memory_result.matches
                else "Поиск в decision memory выполнен, совпадающих ранее отклоненных шаблонов не найдено."
            ),
            source=memory_result.source,
        )
    )
    if memory_result.matches:
        memory_comparison = (
            "Текущая конструкция портфеля сопоставлена с ранее отклоненными случаями из decision memory, "
            "чтобы не повторять известные ошибки концентрации и несоответствия рыночному режиму."
        )

    methodology_result = retrieve_rules(
        (
            f"Explain investment recommendation methodology for profile={profile.risk_profile.value}, "
            f"regime={regime.current_regime.value}, selected_assets={','.join(portfolio.selected_assets[:10])}"
        ),
        k=4,
    )
    asset_scores = state.get("asset_scores", {})
    score_summary = []
    for ticker in portfolio.selected_assets:
        score = asset_scores.get(ticker)
        if score is None:
            continue
        score_summary.append(
            {
                "ticker": ticker,
                "overall_score": round(float(score.overall_score), 4),
                "base_model_score": round(float(score.factors.extra_factors.get("base_model_score", 0.0)), 4),
                "news_sentiment": round(float(score.factors.extra_factors.get("news_sentiment", 0.0)), 4),
                "macro_alignment": round(float(score.factors.extra_factors.get("macro_alignment", 0.0)), 4),
            }
        )

    prompt_payload = ExplainabilityInputPayload(
        critic_verdict=critic_report.verdict.value,
        critic_action=critic_report.recommended_action,
        revision_count=int(state.get("revision_count", 0)),
        regime=regime.current_regime.value,
        policy_summaries=(effective_policy.applied_rule_summaries if effective_policy else []),
        methodology_matches=[match.model_dump() for match in methodology_result.matches],
        construction_notes=portfolio.construction_notes,
        uncertainty_notes=portfolio.uncertainty_notes,
        memory_matches=[ref.model_dump() for ref in memory_refs if ref.layer == "decision_memory"],
        selected_assets=portfolio.selected_assets,
        weights=portfolio.weights,
        rationale=portfolio.rationale,
        exclusion_reasons=portfolio.exclusion_reasons,
        score_summary=score_summary,
        backtest_summary=_build_backtest_summary(state),
        process_summary=_build_process_summary(state),
    )

    llm = get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(ExplainabilitySections, method="function_calling")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты старший консультант по управлению капиталом.
Количественная команда уже построила и верифицировала портфель алгоритмически.
Твоя задача — написать понятный клиентский отчет на русском языке и вернуть JSON строго по схеме.

Раздел 1 (executive_summary): связное объяснение в 3 абзацах про макросреду,
сигналы XGBoost, методологические правила и то, почему текущая конструкция портфеля выглядит обоснованной.
Если `critic_verdict` не равен `approve`, НЕ делай вид, что все идеально:
честно объясни, что система протестировала несколько конфигураций, не получила достаточно сильного результата
и поэтому интерпретирует итог как более защитную рекомендацию с акцентом на кэш и осторожность.

Раздел 2 (regime_context): одно предложение, почему именно текущий рыночный режим
привел к такой аллокации активов.

Раздел 3 (risk_disclaimer): стандартный профессиональный дисклеймер о рисках в 2 предложениях.
"""),
        ("human", "Структурированный JSON для explainability:\n{payload_json}")
    ])

    sections: ExplainabilitySections = (prompt | structured_llm).invoke({
        "payload_json": json.dumps(prompt_payload.model_dump(), ensure_ascii=True),
    })

    inclusion_reasons = {
        ticker: f"Включен в портфель, потому что итоговая аллокация присвоила ему материальный вес {portfolio.weights[ticker]:.1%}."
        for ticker in portfolio.selected_assets
    }

    # Assemble the final output fully in Python — portfolio object is injected directly
    final_report = FinalRecommendation(
        portfolio=portfolio,
        executive_summary=sections.executive_summary,
        regime_context=sections.regime_context,
        risk_disclaimer=sections.risk_disclaimer,
        inclusion_reasons=inclusion_reasons,
        exclusion_reasons=portfolio.exclusion_reasons,
        uncertainty_notes=portfolio.uncertainty_notes,
        policy_summary=(effective_policy.applied_rule_summaries if effective_policy else []),
        memory_comparison=memory_comparison,
        backtest_summary=prompt_payload.backtest_summary,
        process_summary=prompt_payload.process_summary,
    )

    logger.info("Generated final client recommendation document.")
    provenance = dict(state.get("provenance", {}))
    provenance["final_report"] = build_provenance_record(
        source="explainability_llm",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.70,
        details={"selected_assets": len(portfolio.selected_assets)},
    )
    provenance["decision_memory_explainability"] = build_provenance_record(
        source=memory_result.source,
        staleness_status=FreshnessStatus.fresh,
        confidence=0.80,
        retrieval_id=memory_result.retrieval_id,
        details={"match_count": len(memory_result.matches)},
    )
    provenance["knowledge_methodology_explainability"] = build_provenance_record(
        source=methodology_result.source,
        staleness_status=FreshnessStatus.fresh,
        confidence=0.86,
        retrieval_id=methodology_result.retrieval_id,
        details={"match_count": len(methodology_result.matches)},
    )
    decision_log = append_decision_log(
        state,
        stage="explainability",
        event="final_report_generated",
        message="Сформирован финальный отчет по результату последней итерации графа.",
        tool_calls=["retrieve_past_mistakes", "retrieve_rules"],
        metadata={
            "selected_assets": len(portfolio.selected_assets),
            "memory_comparison": bool(memory_comparison),
            "knowledge_matches": len(methodology_result.matches),
            "critic_verdict": critic_report.verdict.value,
        },
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["decision_memory"] = FreshnessStatus.fresh.value
    freshness_map["final_report"] = FreshnessStatus.fresh.value
    freshness_map["knowledge_methodology_explainability"] = FreshnessStatus.fresh.value
    memory_refs.append(
        MemoryReference(
            layer="knowledge_base",
            retrieval_id=methodology_result.retrieval_id,
            summary=f"Для explainability stage извлечено {len(methodology_result.matches)} методологических правила.",
            source=methodology_result.source,
        )
    )
    return {
        "final_report": final_report,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "memory_refs": memory_refs,
        "decision_log": decision_log,
    }
