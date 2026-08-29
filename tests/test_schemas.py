from app.domain.schemas import (
    Constraints,
    InvestorProfile,
    MemoryReference,
    RebalancingPolicy,
    RiskProfile,
)
from app.graph.pipeline import build_initial_state


def test_investor_profile_expanded_defaults():
    profile = InvestorProfile(
        risk_profile=RiskProfile.moderate,
        horizon_years=10,
        constraints=Constraints(
            max_asset_weight=0.25,
            max_sector_weight=0.40,
            allowed_asset_classes=["stocks", "bonds", "commodities"],
            forbidden_assets=[],
            max_drawdown_tolerance=0.20,
            min_cash_weight=0.02,
            max_correlation_threshold=0.85,
        ),
    )

    assert profile.investment_amount is None
    assert profile.income_preference is None
    assert profile.sector_restrictions == []
    assert profile.country_restrictions == []
    assert isinstance(profile.rebalancing_policy, RebalancingPolicy)
    assert profile.rebalancing_policy.mode == "threshold_and_periodic"
    assert profile.rebalancing_policy.period_days == 30
    assert profile.rebalancing_policy.drift_threshold == 0.05
    assert profile.rebalancing_policy.review_frequency == "monthly"


def test_initial_graph_state_includes_extended_contract_fields():
    initial_state = build_initial_state("Build a moderate long-term portfolio.")

    assert initial_state["request_id"]
    assert initial_state["correlation_id"]
    assert initial_state["fundamentals"] == {}
    assert initial_state["macro_data"] is None
    assert initial_state["features"] == {}
    assert initial_state["news_digest"] is None
    assert initial_state["freshness_map"] == {}
    assert initial_state["memory_refs"] == []


def test_memory_reference_schema_is_explicit():
    ref = MemoryReference(
        layer="decision_memory",
        retrieval_id="memory-123",
        summary="Prior rejection for concentration risk.",
        source="decision_memory_rag",
    )

    assert ref.layer == "decision_memory"
    assert ref.retrieval_id == "memory-123"
    assert ref.source == "decision_memory_rag"
