from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from app.agents.base import get_llm
from app.domain.schemas import FreshnessStatus, InvestorProfile
from app.graph.telemetry import build_provenance_record
from app.observability.logger import get_logger
from app.graph.state import GraphState
logger = get_logger(__name__)

def run_profile_agent(state: GraphState) -> Dict[str, Any]:
    """
    Profile Agent: Parses the raw user query and strictly outputs the Pydantic InvestorProfile.
    """
    logger.info("Running Profile Agent.")
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(InvestorProfile, method="function_calling")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly analytical Quantitative Finance Profiler.
Your job is to read the user's raw message and explicitly map it to the `InvestorProfile` schema.

STRICT RULES:
1. `risk_profile` must be exactly 'conservative', 'moderate', or 'aggressive'.
2. `horizon_years` must be an integer (e.g., 5, 10, 20).
3. `target` must be a 1-sentence string describing the growth goal.
4. `investment_amount` should be numeric if explicitly provided, otherwise null.
5. `income_preference` should capture income vs growth preference when the user specifies it, otherwise null.
6. `sector_restrictions` and `country_restrictions` must be arrays, default [].
7. `rebalancing_policy` must be populated. If the user gives no preference, use:
   - mode: "threshold_and_periodic"
   - period_days: 30
   - drift_threshold: 0.05
   - review_frequency: "monthly"
8. `constraints` must define mathematical bounds. If the user doesn't specify, use safe defaults:
   - max_asset_weight: 0.25 (25% max per asset)
   - max_sector_weight: 0.40 (40% max per sector)
   - allowed_asset_classes: ["stocks", "bonds", "commodities"]
   - forbidden_assets: empty list unless user specifies "no crypto" or "no tech"
   - max_drawdown_tolerance: 0.10 for conservative, 0.20 for moderate, 0.35 for aggressive.
   - min_cash_weight: 0.02
   - max_correlation_threshold: 0.85

Output ONLY the strictly validated JSON matching the schema."""),
        ("human", "User Request: {user_query}")
    ])
    
    chain = prompt | structured_llm
    
    profile: InvestorProfile = chain.invoke({"user_query": state["user_query"]})
    
    logger.info(
        "Identified investor profile=%s horizon_years=%s.",
        profile.risk_profile.value,
        profile.horizon_years,
    )

    provenance = dict(state.get("provenance", {}))
    provenance["profile"] = build_provenance_record(
        source="profile_llm_parser",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.78,
        details={
            "risk_profile": profile.risk_profile.value,
            "horizon_years": profile.horizon_years,
        },
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["profile"] = FreshnessStatus.fresh.value

    return {
        "profile": profile,
        "provenance": provenance,
        "freshness_map": freshness_map,
    }
