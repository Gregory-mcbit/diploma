from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.domain.schemas import (
    AssetScore,
    BacktestResult,
    CandidatePortfolio,
    CriticReport,
    DecisionLogEntry,
    EffectivePolicy,
    FinalRecommendation,
    FeatureSnapshot,
    FundamentalSnapshot,
    InvestorProfile,
    MonitoringDecision,
    NewsDigest,
    ProvenanceRecord,
    RegimeReport,
    RiskReport,
    TraceEvent,
)


class HealthResponse(BaseModel):
    status: str
    service: str


class PortfolioRunRequest(BaseModel):
    user_query: str


class MonitoringRunRequest(BaseModel):
    profile: InvestorProfile
    active_portfolio: CandidatePortfolio
    user_query: str = "monitoring"


class AuditPayload(BaseModel):
    request_id: str
    correlation_id: str
    decision_log: List[DecisionLogEntry] = Field(default_factory=list)
    provenance: Dict[str, ProvenanceRecord] = Field(default_factory=dict)
    freshness_map: Dict[str, str] = Field(default_factory=dict)
    memory_refs: List[Dict[str, Any]] = Field(default_factory=list)
    trace_log: List[TraceEvent] = Field(default_factory=list)


class PortfolioRunResponse(BaseModel):
    final_report: FinalRecommendation
    critic_report: CriticReport
    market_regime: RegimeReport
    backtest_result: BacktestResult
    risk_report: RiskReport
    effective_policy: EffectivePolicy
    profile: InvestorProfile | None = None
    asset_scores: Dict[str, AssetScore] = Field(default_factory=dict)
    fundamentals: Dict[str, FundamentalSnapshot] = Field(default_factory=dict)
    features: Dict[str, FeatureSnapshot] = Field(default_factory=dict)
    news_digest: NewsDigest | None = None
    audit: AuditPayload


class MonitoringRunResponse(BaseModel):
    monitoring_decision: MonitoringDecision
    market_regime: RegimeReport
    risk_report: RiskReport
    effective_policy: EffectivePolicy
    audit: AuditPayload
