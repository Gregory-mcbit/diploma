from typing import Dict, List, Optional
from typing_extensions import TypedDict
from app.domain.schemas import (
    InvestorProfile,
    RegimeReport,
    AssetScore,
    CandidatePortfolio,
    RiskReport,
    CriticReport,
    FinalRecommendation
)


class GraphState(TypedDict):
    """
    The shared state dictionary representing the data flow through our LangGraph multi-agent pipeline.
    This design actively prevents State Bloat by holding refs to massive data structures rather than
    the raw data itself.
    """
    
    # ----------------------------------------
    # Pipeline Inputs
    # ----------------------------------------
    user_query: str
    
    # ----------------------------------------
    # 1. Profile Phase
    # ----------------------------------------
    profile: Optional[InvestorProfile]
    
    # ----------------------------------------
    # 2. Market Environment & Data Phase
    # ----------------------------------------
    universe: List[str]
    regime: Optional[RegimeReport]
    asset_scores: Dict[str, AssetScore]
    
    # Storage/DB References (Used to avoid stuffing JSONs with heavy dataframes)
    market_data_ref: Optional[str]
    news_context_ref: Optional[str]
    
    # ----------------------------------------
    # 3. Validation & Critic Loop Phase
    # ----------------------------------------
    current_portfolio: Optional[CandidatePortfolio]
    risk_report: Optional[RiskReport]
    
    # CRITICAL: List of past critic reports to prevent infinite loops (The circuit breaker memory)
    critic_history: List[CriticReport] 
    revision_count: int
    
    # ----------------------------------------
    # 4. Final Output Phase
    # ----------------------------------------
    final_recommendation: Optional[FinalRecommendation]
