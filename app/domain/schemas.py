from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel, Field


# ==========================================
# ENUMS
# ==========================================


class RiskProfile(str, Enum):
    conservative = "conservative"
    moderate = "moderate"
    aggressive = "aggressive"


class RegimeType(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    sideways = "sideways"
    high_volatility = "high_volatility"
    risk_on = "risk_on"
    risk_off = "risk_off"


class CriticVerdict(str, Enum):
    approve = "approve"
    revise_weights = "revise_weights"
    replace_assets = "replace_assets"
    reduce_risk = "reduce_risk"
    insufficient_confidence = "insufficient_confidence"


class FreshnessStatus(str, Enum):
    fresh = "fresh"
    stale = "stale"
    unknown = "unknown"


class MonitoringAction(str, Enum):
    hold = "hold"
    rebalance_now = "rebalance_now"
    reduce_risk = "reduce_risk"
    reassess_universe = "reassess_universe"
    escalate_manual_review = "escalate_manual_review"


# ==========================================
# 1. PROFILE & CONSTRAINTS
# ==========================================


class Constraints(BaseModel):
    max_asset_weight: float
    max_sector_weight: float
    allowed_asset_classes: List[str]
    forbidden_assets: List[str]
    max_drawdown_tolerance: float
    min_cash_weight: float = 0.0
    max_correlation_threshold: float = 0.85


class RebalancingPolicy(BaseModel):
    mode: str = "threshold_and_periodic"
    period_days: int = 30
    drift_threshold: float = 0.05
    review_frequency: str = "monthly"


class InvestorProfile(BaseModel):
    risk_profile: RiskProfile
    horizon_years: int
    target: str = "Target not explicitly defined."
    investment_amount: float | None = None
    income_preference: str | None = None
    sector_restrictions: List[str] = Field(default_factory=list)
    country_restrictions: List[str] = Field(default_factory=list)
    constraints: Constraints
    rebalancing_policy: RebalancingPolicy = Field(default_factory=RebalancingPolicy)


class PolicyRule(BaseModel):
    id: str
    category: str
    description: str
    risk_profiles: List[RiskProfile] = Field(default_factory=list)
    regimes: List[RegimeType] = Field(default_factory=list)
    max_asset_weight: float | None = None
    max_sector_weight: float | None = None
    min_cash_weight: float | None = None
    max_drawdown_tolerance: float | None = None
    max_correlation_threshold: float | None = None
    allowed_asset_classes: List[str] = Field(default_factory=list)
    forbidden_assets: List[str] = Field(default_factory=list)
    priority: int = 0


class EffectivePolicy(BaseModel):
    constraints: Constraints
    restricted_sectors: List[str] = Field(default_factory=list)
    restricted_countries: List[str] = Field(default_factory=list)
    income_preference: str | None = None
    min_income_weight: float = 0.0
    max_growth_weight: float = 1.0
    applied_rule_ids: List[str] = Field(default_factory=list)
    applied_rule_summaries: List[str] = Field(default_factory=list)
    source: str = "policy_engine"


class ProvenanceRecord(BaseModel):
    source: str
    timestamp: str
    staleness_status: FreshnessStatus
    confidence: float
    retrieval_id: str
    details: Dict[str, Any] = Field(default_factory=dict)


class DecisionLogEntry(BaseModel):
    stage: str
    event: str
    message: str
    timestamp: str
    tool_calls: List[str] = Field(default_factory=list)
    rule_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    request_id: str
    correlation_id: str
    stage: str
    event: str
    message: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTraceRecord(BaseModel):
    request_id: str
    correlation_id: str
    run_type: str
    status: str
    started_at: str
    completed_at: str
    trace_log: List[TraceEvent] = Field(default_factory=list)
    decision_log: List[DecisionLogEntry] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryReference(BaseModel):
    layer: str
    retrieval_id: str
    summary: str
    source: str | None = None


class FundamentalSnapshot(BaseModel):
    ticker: str
    metrics: Dict[str, float | None] = Field(default_factory=dict)
    source: str | None = None


class FeatureSnapshot(BaseModel):
    ticker: str
    values: Dict[str, float] = Field(default_factory=dict)


class MacroData(BaseModel):
    values: Dict[str, float] = Field(default_factory=dict)
    source: str | None = None


class NewsArticle(BaseModel):
    ticker: str
    source: str
    timestamp: str
    title: str
    summary: str
    url: str
    sentiment_tag: str | None = None


class NewsDigestItem(BaseModel):
    ticker: str
    article_count: int
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    titles: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class NewsDigest(BaseModel):
    article_count: int = 0
    tickers: List[str] = Field(default_factory=list)
    sentiment_totals: Dict[str, int] = Field(default_factory=dict)
    by_ticker: Dict[str, NewsDigestItem] = Field(default_factory=dict)


# ==========================================
# 2. MARKET ENVIRONMENT & SCORING
# ==========================================


class RegimeReport(BaseModel):
    current_regime: RegimeType
    confidence: float
    drivers: List[str]
    is_risk_off: bool
    signal_components: Dict[str, float] = Field(default_factory=dict)
    uncertainty_notes: List[str] = Field(default_factory=list)


class RegimeModelOutput(BaseModel):
    regime: RegimeType
    confidence: float
    is_risk_off: bool
    signal_components: Dict[str, float] = Field(default_factory=dict)
    uncertainty_notes: List[str] = Field(default_factory=list)


class FactorScores(BaseModel):
    # Core baseline metrics that should ideally always be calculated
    momentum: float
    volatility: float
    quality: float
    # Flexible container for any additional or experimental ML features
    extra_factors: Dict[str, float] = Field(default_factory=dict)


class AssetScore(BaseModel):
    asset_ticker: str
    factors: FactorScores
    overall_score: float
    confidence: float


# ==========================================
# 3. PORTFOLIO & VALIDATION
# ==========================================


class CandidatePortfolio(BaseModel):
    selected_assets: List[str]
    weights: Dict[str, float]
    cash_weight: float
    rationale: List[str]
    allocation_mode: str = "mean_variance_optimizer"
    construction_notes: List[str] = Field(default_factory=list)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    uncertainty_notes: List[str] = Field(default_factory=list)


class RiskReport(BaseModel):
    portfolio_volatility: float
    max_drawdown_estimate: float
    avg_correlation: float
    concentration_hhi: float = 0.0
    var_95: float = 0.0
    violations: List[str]  # Hard constraint breaches mapped to strings
    warnings: List[str]    # Soft issues flagged by Risk Agent
    fit_to_profile: str


class CriticReport(BaseModel):
    verdict: CriticVerdict
    issues: List[str]
    recommended_action: str


# ==========================================
# 4. FINAL OUTPUT
# ==========================================


class FinalRecommendation(BaseModel):
    portfolio: CandidatePortfolio
    executive_summary: str
    regime_context: str
    risk_disclaimer: str
    inclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    uncertainty_notes: List[str] = Field(default_factory=list)
    policy_summary: List[str] = Field(default_factory=list)
    memory_comparison: str | None = None
    backtest_summary: List[str] = Field(default_factory=list)
    process_summary: List[str] = Field(default_factory=list)


class BacktestResult(BaseModel):
    portfolio_total_return: float
    benchmark_total_return: float
    equal_weight_total_return: float
    portfolio_geometric_mean_return: float = 0.0
    portfolio_volatility: float
    portfolio_max_drawdown: float
    turnover: float
    observations: int
    curve_dates: List[str] = Field(default_factory=list)
    portfolio_curve: List[float] = Field(default_factory=list)
    benchmark_curve: List[float] = Field(default_factory=list)
    equal_weight_curve: List[float] = Field(default_factory=list)
    drawdown_curve: List[float] = Field(default_factory=list)


class MonitoringDecision(BaseModel):
    action: MonitoringAction
    reasons: List[str]
    trigger_flags: List[str]
    summary: str
