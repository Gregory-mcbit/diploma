import pandas as pd
import numpy as np
from typing import Dict
from app.domain.schemas import AssetScore, Constraints
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier


def optimize_portfolio(price_df: pd.DataFrame, scores: Dict[str, AssetScore], constraints: Constraints) -> Dict[str, float]:
    """
    Real quantitative optimization using PyPortfolioOpt.
    Translates XGBoost alpha predictions ('overall_score') and empirical covariance 
    into rigorously bounded weights.
    """
    print("[TOOL] Running PyPortfolioOpt Efficient Frontier bounds enforcement...")
    tickers = list(scores.keys())
    
    # Clean data structure
    clean_prices = price_df[tickers].dropna(how="all")
    
    # 1. Covariance Matrix via Ledoit-Wolf shrinkage (industry standard robustness)
    # This prevents the optimizer from doing crazy things with minor correlations
    S = risk_models.CovarianceShrinkage(clean_prices).ledoit_wolf()
    
    # 2. Expected Returns vector from ML predictions
    # Note: ML scores are predicted 21-day returns. We map them directly to expected return.
    mu_dict = {ticker: scores[ticker].overall_score for ticker in tickers}
    mu = pd.Series(mu_dict)
    
    # 3. Efficient Frontier setup
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, constraints.max_asset_weight))
    
    try:
        # We try to maximize Sharpe ratio conceptually, but if ML scores are negative it may fail
        ef.max_sharpe(risk_free_rate=0.0)
    except Exception:
        try:
            # Fallback for Edge cases: minimize volatility subject to expected returns
            ef.min_volatility()
        except Exception:
            # If all fails, return equal weights bounded by constraints
            n = len(tickers)
            cap = constraints.max_asset_weight
            eq = min(1.0 / n, cap)
            return {t: eq for t in tickers}
            
    cleaned_weights = ef.clean_weights()
    return dict(cleaned_weights)
