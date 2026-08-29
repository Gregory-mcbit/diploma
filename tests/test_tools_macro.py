import pandas as pd

from app.tools.macro_data import _coerce_close_to_series, fetch_macro_snapshot


def test_fetch_macro_snapshot(monkeypatch):
    def fake_series(values):
        return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))

    series_map = {
        "^VIX": fake_series([18 + i * 0.1 for i in range(30)]),
        "^TNX": fake_series([4.0 + i * 0.01 for i in range(30)]),
        "^IRX": fake_series([3.5 + i * 0.01 for i in range(30)]),
        "HYG": fake_series([100 + i * 0.2 for i in range(30)]),
        "LQD": fake_series([100 + i * 0.05 for i in range(30)]),
        "UUP": fake_series([30 + i * 0.05 for i in range(30)]),
        "CL=F": fake_series([70 + i * 0.4 for i in range(30)]),
        "SPY": fake_series([400 + i * 1.0 for i in range(30)]),
        "TLT": fake_series([95 + i * 0.2 for i in range(30)]),
    }

    monkeypatch.setattr(
        "app.tools.macro_data._download_close_series",
        lambda ticker, period="35d": series_map[ticker],
    )

    snapshot = fetch_macro_snapshot()

    assert snapshot.source == "yfinance.macro_snapshot"
    assert set(snapshot.values.keys()) == {
        "vix",
        "yield_10y",
        "yield_3m",
        "yield_spread",
        "credit_spread",
        "usd_strength",
        "oil_mom_1m",
        "spy_tlt_ratio",
    }
    assert snapshot.values["yield_spread"] > 0


def test_coerce_close_to_series_accepts_single_column_dataframe():
    frame = pd.DataFrame(
        {"HYG": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    series = _coerce_close_to_series(frame, "HYG")

    assert isinstance(series, pd.Series)
    assert float(series.iloc[-1]) == 102.0
