import pandas as pd
import numpy as np
from typing import Dict
from app.domain.schemas import AssetScore, Constraints
from app.domain.asset_metadata import ASSET_SECTOR_MAP
from app.observability.logger import get_logger
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier

logger = get_logger(__name__)


def _invested_cap(constraints: Constraints) -> float:
    invested_cap = 1.0 - constraints.min_cash_weight
    if invested_cap <= 0.0:
        raise RuntimeError("Portfolio constraints are infeasible: min_cash_weight must be less than 100%.")
    return invested_cap


def _effective_upper_bound(constraints: Constraints) -> float:
    invested_cap = _invested_cap(constraints)
    effective_upper = constraints.max_asset_weight / invested_cap
    if effective_upper <= 0.0:
        raise RuntimeError("Portfolio constraints are infeasible: max_asset_weight must be positive.")
    return min(1.0, effective_upper)


def _assert_optimizer_feasibility(asset_count: int, constraints: Constraints) -> None:
    if asset_count <= 0:
        raise RuntimeError("Portfolio optimization requires at least one investable asset.")
    effective_upper = _effective_upper_bound(constraints)
    if asset_count * effective_upper + 1e-8 < 1.0:
        invested_cap = _invested_cap(constraints)
        raise RuntimeError(
            "Portfolio constraints are infeasible for the investable universe: "
            f"{asset_count} assets cannot satisfy max_asset_weight={constraints.max_asset_weight:.0%} "
            f"with min_cash_weight={constraints.min_cash_weight:.0%}. "
            f"Available risky allocation is {invested_cap:.0%}."
        )


def _cap_sector_exposure(weights: Dict[str, float], max_sector_weight: float) -> Dict[str, float]:
    if not weights or max_sector_weight >= 1.0:
        return dict(weights)

    adjusted = dict(weights)
    sector_totals: Dict[str, float] = {}
    for ticker, weight in adjusted.items():
        sector = ASSET_SECTOR_MAP.get(ticker, ticker)
        sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

    for sector, total in list(sector_totals.items()):
        if total <= max_sector_weight:
            continue
        scale = max_sector_weight / total
        for ticker, weight in list(adjusted.items()):
            if ASSET_SECTOR_MAP.get(ticker, ticker) == sector:
                adjusted[ticker] = weight * scale

    return adjusted


def _enforce_min_cash(weights: Dict[str, float], min_cash_weight: float) -> Dict[str, float]:
    if not weights or min_cash_weight <= 0.0:
        return dict(weights)

    invested_cap = max(0.0, 1.0 - min_cash_weight)
    invested = sum(weights.values())
    if invested <= invested_cap or invested <= 0.0:
        return dict(weights)

    scale = invested_cap / invested
    return {ticker: weight * scale for ticker, weight in weights.items()}


def _coerce_score(score: float | AssetScore) -> float:
    if isinstance(score, AssetScore):
        return float(score.overall_score)
    return float(score)


def _renormalize_with_cap(weights: Dict[str, float], max_asset_weight: float) -> Dict[str, float]:
    adjusted = dict(weights)
    remaining = 1.0
    free = set(adjusted.keys())

    while free:
        subtotal = sum(adjusted[ticker] for ticker in free)
        if subtotal <= 0:
            break
        capped = False
        for ticker in list(free):
            scaled = adjusted[ticker] / subtotal * remaining
            if scaled > max_asset_weight:
                adjusted[ticker] = max_asset_weight
                remaining -= max_asset_weight
                free.remove(ticker)
                capped = True
        if not capped:
            for ticker in free:
                adjusted[ticker] = adjusted[ticker] / subtotal * remaining
            break
    return adjusted


def rank_allocate_portfolio(
    scores: Dict[str, float | AssetScore],
    constraints: Constraints,
) -> Dict[str, float]:
    if not scores:
        raise ValueError("rank_allocate_portfolio requires non-empty scores.")

    ranked = sorted(scores.items(), key=lambda item: _coerce_score(item[1]), reverse=True)
    max_assets = max(1, int((1.0 - constraints.min_cash_weight) / max(constraints.max_asset_weight, 1e-6)))
    selected = ranked[:max_assets]
    rank_weights = {ticker: float(len(selected) - idx) for idx, (ticker, _) in enumerate(selected)}
    rank_weights = _renormalize_with_cap(rank_weights, constraints.max_asset_weight)
    rank_weights = _cap_sector_exposure(rank_weights, constraints.max_sector_weight)
    rank_weights = _enforce_min_cash(rank_weights, constraints.min_cash_weight)
    return {ticker: float(weight) for ticker, weight in rank_weights.items() if weight > 1e-6}


def optimize_portfolio(
    price_df: pd.DataFrame,
    scores: Dict[str, float | AssetScore],
    constraints: Constraints,
) -> Dict[str, float]:
    """
    Real quantitative optimization using PyPortfolioOpt.
    Translates model-predicted forward returns and empirical covariance into
    rigorously bounded weights, then applies deterministic policy caps.
    """
    logger.info("Running PyPortfolioOpt Efficient Frontier bounds enforcement.")
    tickers = list(scores.keys())
    invested_cap = _invested_cap(constraints)
    _assert_optimizer_feasibility(len(tickers), constraints)
    
    # Clean data structure
    clean_prices = price_df[tickers].dropna(how="all")
    
    # 1. Covariance Matrix via Ledoit-Wolf shrinkage (industry standard robustness)
    # This prevents the optimizer from doing crazy things with minor correlations
    S = risk_models.CovarianceShrinkage(clean_prices).ledoit_wolf()
    
    # 2. Expected Returns vector from ML predictions
    # Note: ML scores are predicted 21-day returns. We map them directly to expected return.
    mu_dict = {ticker: _coerce_score(scores[ticker]) for ticker in tickers}
    mu = pd.Series(mu_dict)
    
    # 3. Efficient Frontier setup
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, _effective_upper_bound(constraints)))
    
    try:
        if float(mu.max()) > 0:
            ef.max_sharpe(risk_free_rate=0.0)
        else:
            ef.min_volatility()
    except Exception as e:
        raise RuntimeError(f"Portfolio optimization failed: {e}") from e
            
    cleaned_weights = {
        ticker: float(weight) * invested_cap
        for ticker, weight in dict(ef.clean_weights()).items()
    }
    cleaned_weights = _cap_sector_exposure(cleaned_weights, constraints.max_sector_weight)
    cleaned_weights = _enforce_min_cash(cleaned_weights, constraints.min_cash_weight)
    return {
        ticker: float(weight)
        for ticker, weight in cleaned_weights.items()
        if weight > 1e-6
    }
