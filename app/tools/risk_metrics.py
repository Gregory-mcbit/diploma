import numpy as np
import pandas as pd
from typing import Dict, List
from app.domain.asset_metadata import ASSET_COUNTRY_MAP, ASSET_SECTOR_MAP, ASSET_STYLE_MAP
from app.domain.schemas import Constraints, EffectivePolicy, RiskReport


INCOME_STYLES = {"income_defensive", "cyclical_income"}
GROWTH_STYLES = {"growth", "core_growth"}


def calculate_portfolio_volatility(weights: Dict[str, float], cov_matrix: pd.DataFrame) -> float:
    """Matrix multiplication w^T * Cov * w"""
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    port_var = np.dot(w.T, np.dot(cov_matrix.loc[tickers, tickers], w))
    return float(np.sqrt(port_var) * np.sqrt(252))


def calculate_concentration_hhi(weights: Dict[str, float]) -> float:
    return float(sum(float(weight) ** 2 for weight in weights.values()))


def calculate_var_95(portfolio_returns: pd.Series) -> float:
    if portfolio_returns.empty:
        return 0.0
    return float(max(0.0, -portfolio_returns.quantile(0.05)))


def calculate_average_pairwise_correlation(returns: pd.DataFrame) -> float:
    if returns.shape[1] < 2:
        return 0.0
    corr = returns.corr().replace([np.inf, -np.inf], np.nan)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    values = corr.where(mask).stack().dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def check_concentration_limits(weights: Dict[str, float], max_weight: float) -> List[str]:
    """Hard deterministic rule checks that an LLM cannot argue with."""
    violations = []
    for ticker, w in weights.items():
        if w > max_weight + 0.001:  # float tolerance
            violations.append(f"Жесткое нарушение лимита: вес {ticker} = {w:.2%}, что выше максимума {max_weight:.2%}")
    return violations


def check_sector_limits(weights: Dict[str, float], max_sector_weight: float) -> List[str]:
    violations = []
    sector_totals: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = ASSET_SECTOR_MAP.get(ticker, ticker)
        sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

    for sector, total in sector_totals.items():
        if total > max_sector_weight + 0.001:
            violations.append(
                f"Нарушение секторной концентрации: вес сектора {sector} = {total:.2%}, что выше максимума {max_sector_weight:.2%}"
            )
    return violations


def check_restricted_policy_exposures(
    weights: Dict[str, float],
    restricted_sectors: List[str],
    restricted_countries: List[str],
) -> List[str]:
    violations = []
    normalized_sectors = {sector.lower() for sector in restricted_sectors}
    normalized_countries = {country.lower() for country in restricted_countries}

    for ticker, weight in weights.items():
        if weight <= 0.0:
            continue
        sector = ASSET_SECTOR_MAP.get(ticker, "unknown").lower()
        country = ASSET_COUNTRY_MAP.get(ticker, "unknown").lower()
        if normalized_sectors and sector in normalized_sectors:
            violations.append(
                f"Нарушение ограничения по сектору: {ticker} относится к сектору '{sector}' с весом {weight:.2%}"
            )
        if normalized_countries and country in normalized_countries:
            violations.append(
                f"Нарушение ограничения по стране: {ticker} относится к стране '{country}' с весом {weight:.2%}"
            )
    return violations


def check_income_preference_alignment(
    weights: Dict[str, float],
    income_preference: str | None,
    min_income_weight: float,
    max_growth_weight: float,
) -> List[str]:
    if income_preference != "income":
        return []

    income_weight = sum(
        weight for ticker, weight in weights.items()
        if ASSET_STYLE_MAP.get(ticker, "") in INCOME_STYLES
    )
    growth_weight = sum(
        weight for ticker, weight in weights.items()
        if ASSET_STYLE_MAP.get(ticker, "") in GROWTH_STYLES
    )

    violations = []
    if income_weight + 0.001 < min_income_weight:
        violations.append(
            f"Нарушение income preference: доля income-активов {income_weight:.2%} ниже минимума {min_income_weight:.2%}"
        )
    if growth_weight > max_growth_weight + 0.001:
        violations.append(
            f"Нарушение income preference: доля growth-активов {growth_weight:.2%} выше максимума {max_growth_weight:.2%}"
        )
    return violations


def check_cash_buffer(weights: Dict[str, float], min_cash_weight: float) -> List[str]:
    if min_cash_weight <= 0.0:
        return []
    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    if cash_weight + 0.001 < min_cash_weight:
        return [
            f"Нарушение кэш-буфера: доля кэша {cash_weight:.2%} ниже минимума {min_cash_weight:.2%}"
        ]
    return []


def check_correlation_limits(
    weights: Dict[str, float],
    returns: pd.DataFrame,
    max_correlation_threshold: float,
) -> List[str]:
    if max_correlation_threshold >= 1.0 or returns.empty:
        return []

    corr = returns.corr().fillna(0.0)
    tickers = list(weights.keys())
    violations = []
    for i, left in enumerate(tickers):
        for right in tickers[i + 1:]:
            if left not in corr.index or right not in corr.columns:
                continue
            pair_corr = float(corr.loc[left, right])
            if pair_corr > max_correlation_threshold:
                combined_weight = weights.get(left, 0.0) + weights.get(right, 0.0)
                violations.append(
                    f"Нарушение корреляционного лимита: корреляция {left}/{right} = {pair_corr:.2f}, "
                    f"что выше максимума {max_correlation_threshold:.2f}, при совокупном весе {combined_weight:.2%}"
                )
    return violations


def generate_risk_report(
    weights: Dict[str, float],
    price_df: pd.DataFrame,
    max_asset_weight: float | None = None,
    constraints: Constraints | None = None,
    effective_policy: EffectivePolicy | None = None,
) -> RiskReport:
    """
    Pure mathematical risk validation computing historical volatility, max drawdown,
    concentration, sector caps, cash policy, and pairwise correlation breaches.
    """
    effective_constraints = (
        effective_policy.constraints
        if effective_policy is not None
        else constraints
    ) or Constraints(
        max_asset_weight=max_asset_weight if max_asset_weight is not None else 1.0,
        max_sector_weight=1.0,
        allowed_asset_classes=[],
        forbidden_assets=[],
        max_drawdown_tolerance=1.0,
    )

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
    concentration_hhi = calculate_concentration_hhi(weights)
    var_95 = calculate_var_95(port_returns)
    avg_correlation = calculate_average_pairwise_correlation(returns)
    
    violations = []
    violations.extend(check_concentration_limits(weights, effective_constraints.max_asset_weight))
    violations.extend(check_sector_limits(weights, effective_constraints.max_sector_weight))
    violations.extend(check_cash_buffer(weights, effective_constraints.min_cash_weight))
    violations.extend(
        check_correlation_limits(
            weights=weights,
            returns=returns,
            max_correlation_threshold=effective_constraints.max_correlation_threshold,
        )
    )
    if effective_policy is not None:
        violations.extend(
            check_restricted_policy_exposures(
                weights=weights,
                restricted_sectors=effective_policy.restricted_sectors,
                restricted_countries=effective_policy.restricted_countries,
            )
        )
        violations.extend(
            check_income_preference_alignment(
                weights=weights,
                income_preference=effective_policy.income_preference,
                min_income_weight=effective_policy.min_income_weight,
                max_growth_weight=effective_policy.max_growth_weight,
            )
        )
    
    warnings = []
    if max_dd > 0.20:
        warnings.append(f"Историческая максимальная просадка остается высокой ({max_dd:.2%}).")
    if vol > 0.25:
        warnings.append(f"Волатильность портфеля остается очень высокой ({vol:.2%}).")
    if max_dd > effective_constraints.max_drawdown_tolerance:
        warnings.append(
            f"Просадка выше целевого лимита профиля: максимальная просадка {max_dd:.2%} превышает "
            f"ориентир {effective_constraints.max_drawdown_tolerance:.2%}. Пока это не блокирует рекомендацию, "
            "но должно учитываться как мягкий риск-фактор."
        )
        
    return RiskReport(
        portfolio_volatility=vol,
        max_drawdown_estimate=max_dd,
        avg_correlation=avg_correlation,
        concentration_hhi=concentration_hhi,
        var_95=var_95,
        violations=violations,
        warnings=warnings,
        fit_to_profile="допустимо" if not violations else "нарушение"
    )
