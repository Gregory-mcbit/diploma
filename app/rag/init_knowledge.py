from app.rag.schemas import KnowledgeDocument
from app.rag.knowledge_rag import add_knowledge_rules
import uuid

def initialize_knowledge_base():
    docs = [
        # Original 5 
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="profile_mapping",
            content="The Conservative profile prioritizes capital preservation. Maximum equity exposure is strictly capped at 40%. The remaining 60%+ must be allocated to Fixed Income, Cash, or Gold (e.g., TLT, GLD). Maximum theoretical drawdown tolerance is 10%."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="profile_mapping",
            content="The Moderate profile balances growth and income. Equity exposure should range between 40% and 70%. Fixed income and safe-havens should make up 30% to 60%. Maximum theoretical drawdown tolerance is 20%."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="profile_mapping",
            content="The Aggressive profile seeks long-term capital appreciation and accepts high volatility. Equity exposure can range from 70% up to 100%. Fixed income is optional (0-30%). Maximum theoretical drawdown tolerance is 35%."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="regime_tactical",
            content="In a Bull Market regime (low volatility, positive economic growth), the optimizer should prioritize assets with high Momentum scores (1M and 3M momentum). Concentration limits can be relaxed to allow up to 25% allocation to a single high-performing sector or stock."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="regime_tactical",
            content="In a Bear Market or High-Stress transition regime (high VIX, negative momentum), the system must reduce overall equity exposure by 15-20% relative to the baseline profile. Priority must shift to Quality factors, low-volatility assets, bonds (TLT), and gold (GLD). Single asset concentration must not exceed 10%."
        ),
        # Step 3.6 Explanatory Density
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="concentration_limits",
            content="To prevent idiosyncratic risk, no single asset sector or closely correlated cluster (e.g., Technology via XLK or QQQ) can exceed 35% of the total portfolio weight, regardless of the user's aggressive profile or how bullish the regime is."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="liquidity_rules",
            content="All portfolios must maintain a minimum cash buffer of 2% to absorb transaction costs and rebalancing friction. In high volatility (high VIX) or risk-off transition regimes, the cash buffer must be proactively increased to minimum 10% to preserve directional optionality."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="correlation_rules",
            content="A robust portfolio cannot consist solely of highly correlated assets even if they all have high XGBoost alpha scores. The selection process must penalize or bound assets that historically exhibit >0.85 correlation with the rest of the portfolio."
        ),
        KnowledgeDocument(
            id=str(uuid.uuid4()),
            category="asset_classification",
            content="Equities include SPY, QQQ, AAPL, MSFT. Fixed Income includes TLT, SHY, BND. Commodities/Alternatives include GLD, SLV."
        )
    ]
    
    print("[INIT] Injecting Institutional Knowledge Rules into ChromaDB...")
    add_knowledge_rules(docs)
    print("[INIT] Done! The Knowledge RAG is densely populated and ready for Agent retrieval.")

if __name__ == "__main__":
    initialize_knowledge_base()
