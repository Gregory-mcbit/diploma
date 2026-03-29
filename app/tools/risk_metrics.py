import numpy as np
import pandas as pd
from typing import Dict, List
from app.domain.schemas import RiskReport


def calculate_portfolio_volatility(weights: Dict[str, float], cov_matrix: pd.DataFrame) -> float:
    """Matrix multiplication w^T * Cov * w"""
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    port_var = np.dot(w.T, np.dot(cov_matrix.loc[tickers, tickers], w))
    return float(np.sqrt(port_var) * np.sqrt(252))


def check_concentration_limits(weights: Dict[str, float], max_weight: float) -> List[str]:
    """Hard deterministic rule checks that an LLM cannot argue with."""
    violations = []
    for ticker, w in weights.items():
        if w > max_weight + 0.001:  # float tolerance
            violations.append(f"Hard constraint breach: {ticker} weight {w:.2%} exceeds max {max_weight:.2%}")
    return violations


def generate_risk_report(weights: Dict[str, float], price_df: pd.DataFrame, max_asset_weight: float) -> RiskReport:
    """
    Pure mathematical risk validation computing historical volatility, max drawdown, and concentration.
    """
    tickers = list(weights.keys())
    try:
        returns = price_df[tickers].pct_change().dropna(how='all')
    except KeyError:
        # Handle cases where price_df doesn't have exact tickers
        valid = [t for t in tickers if t in price_df.columns]
        returns = price_df[valid].pct_change().dropna(how='all')
        
    cov = returns.cov()
    
    vol = calculate_portfolio_volatility(weights, cov)
    
    # Synthesize Portfolio returns for Drawdown
    port_returns = returns.dot(pd.Series(weights))
    cumulative = (1 + port_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    
    if len(drawdown) > 0:
        max_dd = float(abs(drawdown.min()))
    else:
        max_dd = 0.0
    
    violations = check_concentration_limits(weights, max_asset_weight)
    
    warnings = []
    if max_dd > 0.20:
        warnings.append(f"Historical Max Drawdown is high ({max_dd:.2%}).")
    if vol > 0.25:
        warnings.append(f"Portfolio Volatility is very high ({vol:.2%}).")
        
    return RiskReport(
        portfolio_volatility=vol,
        max_drawdown_estimate=max_dd,
        avg_correlation=float(cov.mean().mean()),
        violations=violations,
        warnings=warnings,
        fit_to_profile="acceptable" if not violations else "violation"
    )
