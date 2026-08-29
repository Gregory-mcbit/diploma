from __future__ import annotations

import pandas as pd

from app.domain.schemas import BacktestResult


def _max_drawdown(cumulative_returns: pd.Series) -> float:
    rolling_peak = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_peak - 1.0
    return abs(float(drawdown.min()))


def _drawdown_curve(cumulative_returns: pd.Series) -> pd.Series:
    rolling_peak = cumulative_returns.cummax()
    return cumulative_returns / rolling_peak - 1.0


def _series_to_float_list(series: pd.Series) -> list[float]:
    return [float(value) for value in series.astype(float).tolist()]


def _annualized_geometric_return(total_return: float, observations: int) -> float:
    if observations <= 0 or total_return <= -1.0:
        return 0.0
    return float((1.0 + total_return) ** (252.0 / observations) - 1.0)


def backtest_portfolio(
    price_df: pd.DataFrame,
    weights: dict[str, float],
    cash_weight: float = 0.0,
    benchmark_ticker: str = "SPY",
    previous_weights: dict[str, float] | None = None,
) -> BacktestResult:
    tickers = [ticker for ticker in weights if ticker in price_df.columns]
    if not tickers:
        raise ValueError("Backtest requires at least one weighted ticker present in price_df.")
    if cash_weight < 0.0:
        raise ValueError("Backtest cash_weight cannot be negative.")

    clean_prices = price_df[tickers].dropna(how="any")
    if clean_prices.empty:
        raise ValueError("Backtest price frame is empty after aligning selected tickers.")

    returns = clean_prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Backtest requires at least two observations to compute returns.")

    weight_series = pd.Series({ticker: float(weights[ticker]) for ticker in tickers})
    invested_weight = float(weight_series.sum())
    if invested_weight <= 0.0:
        raise ValueError("Backtest requires positive risky asset weights.")
    if invested_weight + float(cash_weight) > 1.000001:
        raise ValueError("Backtest received portfolio weights exceeding 100% gross allocation.")
    portfolio_returns = returns.mul(weight_series, axis=1).sum(axis=1)
    portfolio_curve = (1 + portfolio_returns).cumprod()

    equal_weight_series = pd.Series(1.0 / len(tickers), index=tickers)
    equal_weight_returns = returns.mul(equal_weight_series, axis=1).sum(axis=1)
    equal_weight_curve = (1 + equal_weight_returns).cumprod()

    if benchmark_ticker in price_df.columns:
        benchmark_prices = price_df[benchmark_ticker].reindex(clean_prices.index).ffill().bfill().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna().reindex(returns.index).dropna()
        benchmark_curve = (1 + benchmark_returns).cumprod()
        benchmark_total_return = float(benchmark_curve.iloc[-1] - 1.0)
    else:
        benchmark_curve = equal_weight_curve.copy()
        benchmark_total_return = float(equal_weight_curve.iloc[-1] - 1.0)

    curve_index = returns.index.intersection(benchmark_curve.index).intersection(equal_weight_curve.index)
    portfolio_curve = portfolio_curve.reindex(curve_index)
    benchmark_curve = benchmark_curve.reindex(curve_index)
    equal_weight_curve = equal_weight_curve.reindex(curve_index)
    drawdown_curve = _drawdown_curve(portfolio_curve)
    portfolio_total_return = float(portfolio_curve.iloc[-1] - 1.0)

    turnover = 0.0
    if previous_weights:
        turnover = 0.5 * sum(
            abs(float(weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0)))
            for ticker in set(weights) | set(previous_weights)
        )

    return BacktestResult(
        portfolio_total_return=portfolio_total_return,
        benchmark_total_return=benchmark_total_return,
        equal_weight_total_return=float(equal_weight_curve.iloc[-1] - 1.0),
        portfolio_geometric_mean_return=_annualized_geometric_return(portfolio_total_return, len(curve_index)),
        portfolio_volatility=float(portfolio_returns.std() * (252 ** 0.5)),
        portfolio_max_drawdown=_max_drawdown(portfolio_curve),
        turnover=float(turnover),
        observations=int(len(returns)),
        curve_dates=[str(index.date()) for index in curve_index],
        portfolio_curve=_series_to_float_list(portfolio_curve),
        benchmark_curve=_series_to_float_list(benchmark_curve),
        equal_weight_curve=_series_to_float_list(equal_weight_curve),
        drawdown_curve=_series_to_float_list(drawdown_curve),
    )

