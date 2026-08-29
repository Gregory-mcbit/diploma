from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from app.domain.schemas import (
    BacktestResult,
    CandidatePortfolio,
    CriticReport,
    DecisionLogEntry,
    EffectivePolicy,
    AssetScore,
    FeatureSnapshot,
    FinalRecommendation,
    FundamentalSnapshot,
    InvestorProfile,
    MacroData,
    MemoryReference,
    NewsArticle,
    NewsDigest,
    MonitoringDecision,
    ProvenanceRecord,
    RegimeReport,
    RiskReport,
    TraceEvent,
)


class GraphState(TypedDict):
    user_query: str
    request_id: str
    correlation_id: str

    profile: Optional[InvestorProfile]
    current_universe: List[str]
    market_data_pointer: Optional[str]
    fundamentals: Dict[str, FundamentalSnapshot]
    macro_data: Optional[MacroData]
    features: Dict[str, FeatureSnapshot]
    news_articles: Dict[str, List[NewsArticle]]
    news_digest: Optional[NewsDigest]
    asset_scores: Dict[str, AssetScore]
    market_regime: Optional[RegimeReport]
    effective_policy: Optional[EffectivePolicy]
    active_portfolio: Optional[CandidatePortfolio]
    monitoring_decision: Optional[MonitoringDecision]
    proposed_portfolio: Optional[CandidatePortfolio]
    backtest_result: Optional[BacktestResult]
    risk_report: Optional[RiskReport]
    critic_report: Optional[CriticReport]
    critic_history: List[CriticReport] 
    revision_count: int
    provenance: Dict[str, ProvenanceRecord]
    freshness_map: Dict[str, str]
    memory_refs: List[MemoryReference]
    decision_log: List[DecisionLogEntry]
    trace_log: List[TraceEvent]
    final_report: Optional[FinalRecommendation]
