from app.tools.fundamentals import fetch_fundamentals


class DummyTicker:
    def __init__(self, ticker: str):
        self.info = {
            "trailingPE": 24.0 if ticker == "SPY" else 18.0,
            "forwardPE": 21.0 if ticker == "SPY" else 16.0,
            "priceToBook": 4.2 if ticker == "SPY" else 3.1,
            "returnOnEquity": 0.18,
            "profitMargins": 0.14,
            "debtToEquity": 0.9,
        }


def test_fetch_fundamentals(monkeypatch):
    monkeypatch.setattr("app.tools.fundamentals.yf.Ticker", DummyTicker)

    snapshots = fetch_fundamentals(["SPY", "MSFT"])

    assert set(snapshots.keys()) == {"SPY", "MSFT"}
    assert snapshots["SPY"].metrics["pe"] == 24.0
    assert snapshots["MSFT"].metrics["forward_pe"] == 16.0
    assert snapshots["SPY"].source == "yfinance.info"

