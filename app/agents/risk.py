import pandas as pd
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from app.agents.base import get_llm
from app.domain.schemas import FreshnessStatus, RiskReport
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.tools.risk_metrics import generate_risk_report
logger = get_logger(__name__)

def run_risk_agent(state: GraphState) -> Dict[str, Any]:
    """
    Risk Agent: Runs deterministic boundary math (volatility, drawdown, HHI, VaR).
    LLM maps the raw violations into a strict RiskReport string explanations.
    """
    logger.info("Running Risk Agent.")
    
    profile = state.get("profile")
    portfolio = state.get("proposed_portfolio") or state.get("active_portfolio")
    effective_policy = state.get("effective_policy")
    parquet_pointer = state.get("market_data_pointer")
    
    if not profile or not portfolio or not parquet_pointer:
        raise ValueError("Risk Agent requires profile, portfolio, and market_data_pointer.")
        
    # Load the price df from parquet
    price_df = pd.read_parquet(parquet_pointer)
    
    # Execute deterministic math backend
    math_risk_report = generate_risk_report(
        weights=portfolio.weights,
        price_df=price_df,
        constraints=(effective_policy.constraints if effective_policy else profile.constraints),
        effective_policy=effective_policy,
    )
    
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RiskReport, method="function_calling")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты директор по рискам.
Числовой движок уже вернул математические `violations`, `volatility` и `drawdown`.
Твоя задача — строго отразить это в схеме `RiskReport` и добавить качественный комментарий на русском языке.
Если `Math Violations` не пустой, ты ОБЯЗАН перенести их как критические нарушения.
Сгенерируй 1-2 мягких предупреждения `warnings`, если волатильность высокая, но еще в допустимых рамках.
Поле `fit_to_profile` должно быть одной короткой фразой, объясняющей, соответствует ли риск профилю клиента.
"""),
        ("human", "Профиль: {profile}\nМатематическая волатильность: {vol}\nМатематическая просадка: {dd}\nHHI: {hhi}\nVaR95: {var_95}\nСредняя корреляция: {avg_correlation}\nМатематические нарушения: {violations}\n\nСформируй RiskReport.")
    ])
    
    chain = prompt | structured_llm
    
    risk_report: RiskReport = chain.invoke({
        "profile": profile.risk_profile.value,
        "vol": math_risk_report.portfolio_volatility,
        "dd": math_risk_report.max_drawdown_estimate,
        "hhi": math_risk_report.concentration_hhi,
        "var_95": math_risk_report.var_95,
        "avg_correlation": math_risk_report.avg_correlation,
        "violations": math_risk_report.violations
    })
    
    # Override logic to guarantee math isn't overwritten by hallucination
    risk_report.portfolio_volatility = math_risk_report.portfolio_volatility
    risk_report.max_drawdown_estimate = math_risk_report.max_drawdown_estimate
    risk_report.avg_correlation = math_risk_report.avg_correlation
    risk_report.concentration_hhi = math_risk_report.concentration_hhi
    risk_report.var_95 = math_risk_report.var_95
    risk_report.violations = math_risk_report.violations
    
    logger.info(
        "Analyzed risk: volatility=%.4f violations=%s.",
        risk_report.portfolio_volatility,
        len(risk_report.violations),
    )

    provenance = dict(state.get("provenance", {}))
    provenance["risk_report"] = build_provenance_record(
        source="deterministic_risk_engine",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.90,
        details={"violation_count": len(risk_report.violations)},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["risk_report"] = FreshnessStatus.fresh.value
    decision_log = append_decision_log(
        state,
        stage="risk",
        event="risk_validated",
        message=f"Проведена валидация риска портфеля; жестких нарушений: {len(risk_report.violations)}.",
        tool_calls=["generate_risk_report"],
        rule_ids=(effective_policy.applied_rule_ids if effective_policy else []),
        metadata={"violation_count": len(risk_report.violations)},
    )
    return {
        "risk_report": risk_report,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "decision_log": decision_log,
    }
