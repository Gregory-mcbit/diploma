import datetime
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from app.agents.base import get_llm
from app.domain.schemas import FreshnessStatus, MacroData, NewsDigest, RegimeReport
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.ml.regime_model import infer_regime
logger = get_logger(__name__)


class RegimeNarrativeInput(BaseModel):
    as_of_date: str
    regime: str
    vix: float
    credit_spread: float
    usd_strength: float
    oil_mom_1m: float
    spy_tlt_ratio: float
    yield_spread: float
    spy_momentum_1m: float
    uncertainty_notes: list[str] = Field(default_factory=list)
    news_digest: dict = Field(default_factory=dict)


def run_regime_agent(state: GraphState) -> Dict[str, Any]:
    """
    Regime Agent: Combines deterministic macro signals with LLM interpretation.

    Step 1 — Deterministic math: reads asset_scores for SPY momentum and
              uses VIX + 10Y/3M spread to classify regime quantitatively.
    Step 2 — LLM enriches with qualitative narrative and confidence score.
    """
    logger.info("Running Regime Agent.")

    macro_payload = state.get("macro_data")
    features = state.get("features", {})
    news_digest = state.get("news_digest")
    if not macro_payload:
        raise ValueError("Regime Agent requires macro_data from the Scoring Agent.")
    if "SPY" not in features:
        raise ValueError("Regime Agent requires SPY features from the Scoring Agent.")
    if not news_digest:
        raise ValueError("Regime Agent requires news_digest from the Data Agent.")

    macro = macro_payload.values if isinstance(macro_payload, MacroData) else macro_payload
    spy_feature_payload = features["SPY"]
    if isinstance(spy_feature_payload, dict):
        spy_features = spy_feature_payload
    else:
        spy_features = spy_feature_payload.values
    model_output = infer_regime(macro_data=macro, spy_features=spy_features)
    vix = float(model_output.signal_components["vix"])
    spread = float(model_output.signal_components["yield_spread"])
    spy_mom = float(model_output.signal_components["spy_mom_1m"])

    logger.info(
        "Quant signals: VIX=%.1f Spread=%.2f SPY_mom1m=%.2f%%.",
        vix,
        spread,
        spy_mom * 100,
    )
    logger.info("Deterministic regime=%s.", model_output.regime.value)

    # ── Step 2: LLM adds qualitative context and 3 driver strings ────────────
    llm = get_llm(temperature=0.0)
    prompt_payload = RegimeNarrativeInput(
        as_of_date=datetime.datetime.now().strftime("%Y-%m-%d"),
        regime=model_output.regime.value,
        vix=vix,
        credit_spread=float(model_output.signal_components["credit_spread"]),
        usd_strength=float(model_output.signal_components["usd_strength"]),
        oil_mom_1m=float(model_output.signal_components["oil_mom_1m"]),
        spy_tlt_ratio=float(model_output.signal_components["spy_tlt_ratio"]),
        yield_spread=spread,
        spy_momentum_1m=spy_mom,
        uncertainty_notes=model_output.uncertainty_notes,
        news_digest=(
            news_digest.model_dump()
            if isinstance(news_digest, NewsDigest)
            else news_digest
        ),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Chief Macroeconomic Strategist at BlackRock.
You receive real quantitative macro signals and must produce a short qualitative regime context.
Return EXACTLY 3 lines:
Line 1: driver 1 (e.g. 'VIX at 18.5 signals low fear in equity markets')
Line 2: driver 2
Line 3: driver 3
No headings, no extra text."""),
        ("human", "Structured regime context input JSON:\n{payload_json}\n\nWrite the 3 driver lines.")
    ])

    raw_drivers: str = (prompt | llm | StrOutputParser()).invoke({
        "payload_json": json.dumps(prompt_payload.model_dump(), ensure_ascii=True),
    })

    drivers = [d.strip() for d in raw_drivers.strip().split("\n") if d.strip()][:3]
    if len(drivers) != 3:
        raise RuntimeError("Regime Agent expected exactly 3 LLM driver lines.")

    # ── Assemble RegimeReport deterministically ───────────────────────────────
    regime_report = RegimeReport(
        current_regime=model_output.regime,
        confidence=model_output.confidence,
        drivers=drivers,
        is_risk_off=model_output.is_risk_off,
        signal_components=model_output.signal_components,
        uncertainty_notes=model_output.uncertainty_notes,
    )

    logger.info(
        "Final regime=%s confidence=%.2f risk_off=%s.",
        regime_report.current_regime.value,
        regime_report.confidence,
        regime_report.is_risk_off,
    )

    provenance = dict(state.get("provenance", {}))
    if macro and "macro_data" not in provenance:
        provenance["macro_data"] = build_provenance_record(
            source="yfinance.macro_snapshot",
            staleness_status=FreshnessStatus.fresh,
            confidence=0.85,
            details={"signal_count": len(macro)},
        )
    provenance["market_regime"] = build_provenance_record(
        source="deterministic_rule_plus_llm_context",
        staleness_status=FreshnessStatus.fresh,
        confidence=regime_report.confidence,
        details={
            "is_risk_off": regime_report.is_risk_off,
            "signal_components": regime_report.signal_components,
        },
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["market_regime"] = FreshnessStatus.fresh.value
    decision_log = append_decision_log(
        state,
        stage="regime",
        event="regime_classified",
        message=f"Classified regime as {regime_report.current_regime.value}.",
        tool_calls=["infer_regime"],
        metadata={
            "vix": round(vix, 4),
            "yield_spread": round(spread, 4),
            "spy_mom_1m": round(spy_mom, 6),
        },
    )
    return {
        "market_regime": regime_report,
        "macro_data": MacroData(values=macro, source="yfinance.macro_snapshot"),
        "provenance": provenance,
        "freshness_map": freshness_map,
        "decision_log": decision_log,
    }
