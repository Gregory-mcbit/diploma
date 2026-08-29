import pandas as pd
import pytest

from app.tools.backtest import backtest_portfolio


def test_backtest_portfolio_returns_summary():
    dates = pd.date_range("2024-01-01", periods=40)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 1.0 for i in range(len(dates))],
            "TLT": [90 + i * 0.3 for i in range(len(dates))],
            "GLD": [80 + i * 0.4 for i in range(len(dates))],
        },
        index=dates,
    )

    result = backtest_portfolio(
        price_df=prices,
        weights={"SPY": 0.5, "TLT": 0.3, "GLD": 0.2},
        benchmark_ticker="SPY",
        previous_weights={"SPY": 0.4, "TLT": 0.4, "GLD": 0.2},
    )

    assert result.observations > 0
    assert result.portfolio_total_return > 0
    assert result.equal_weight_total_return > 0
    assert result.turnover > 0
    assert result.portfolio_max_drawdown >= 0
    assert len(result.curve_dates) == result.observations
    assert len(result.portfolio_curve) == result.observations
    assert len(result.benchmark_curve) == result.observations
    assert len(result.equal_weight_curve) == result.observations
    assert len(result.drawdown_curve) == result.observations
    assert result.portfolio_curve[-1] - 1.0 == pytest.approx(result.portfolio_total_return)
    assert result.portfolio_geometric_mean_return > 0
    assert min(result.drawdown_curve) <= 0


def test_backtest_respects_cash_weight():
    dates = pd.date_range("2024-01-01", periods=40)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 1.0 for i in range(len(dates))],
            "TLT": [90 + i * 0.2 for i in range(len(dates))],
        },
        index=dates,
    )

    fully_invested = backtest_portfolio(
        price_df=prices,
        weights={"SPY": 0.8, "TLT": 0.2},
        cash_weight=0.0,
        benchmark_ticker="SPY",
    )
    cash_buffered = backtest_portfolio(
        price_df=prices,
        weights={"SPY": 0.4, "TLT": 0.1},
        cash_weight=0.5,
        benchmark_ticker="SPY",
    )

    assert cash_buffered.portfolio_total_return < fully_invested.portfolio_total_return
