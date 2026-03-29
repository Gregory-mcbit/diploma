from enum import Enum
from typing import Dict, List
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


# ==========================================
# 1. PROFILE & CONSTRAINTS
# ==========================================


class Constraints(BaseModel):
    max_asset_weight: float
    max_sector_weight: float
    allowed_asset_classes: List[str]
    forbidden_assets: List[str]
    max_drawdown_tolerance: float


class InvestorProfile(BaseModel):
    risk_profile: RiskProfile
    horizon_years: int
    target: str
    constraints: Constraints
    rebalancing_policy: Dict[str, str]


# ==========================================
# 2. MARKET ENVIRONMENT & SCORING
# ==========================================


class RegimeReport(BaseModel):
    current_regime: RegimeType
    confidence: float
    drivers: List[str]
    is_risk_off: bool


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


class RiskReport(BaseModel):
    portfolio_volatility: float
    max_drawdown_estimate: float
    avg_correlation: float
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
