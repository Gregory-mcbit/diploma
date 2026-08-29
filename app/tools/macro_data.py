from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.domain.schemas import MacroData
from app.observability.logger import get_logger


REQUIRED_MACRO_FIELDS = (
    "vix",
    "yield_10y",
    "yield_3m",
    "yield_spread",
    "credit_spread",
    "usd_strength",
    "oil_mom_1m",
    "spy_tlt_ratio",
)
logger = get_logger(__name__)


def _coerce_close_to_series(close_data: pd.Series | pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(close_data, pd.DataFrame):
        if close_data.shape[1] != 1:
            raise RuntimeError(
                f"Macro data close selection for {ticker} returned {close_data.shape[1]} columns, expected exactly 1."
            )
        close_series = close_data.iloc[:, 0]
    else:
        close_series = close_data

    if not isinstance(close_series, pd.Series):
        raise RuntimeError(f"Macro data close selection for {ticker} did not resolve to a pandas Series.")

    return pd.to_numeric(close_series, errors="raise")


def _download_close_series(ticker: str, period: str = "35d") -> pd.Series:
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError(f"Macro data download returned no rows for {ticker}.")
    close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
    series = _coerce_close_to_series(close, ticker).dropna()
    if series.empty:
        raise RuntimeError(f"Macro data close series is empty for {ticker}.")
    return series


def fetch_macro_snapshot(period: str = "35d") -> MacroData:
    vix_s = _download_close_series("^VIX", period=period)
    tnx_s = _download_close_series("^TNX", period=period)
    irx_s = _download_close_series("^IRX", period=period)
    hyg_s = _download_close_series("HYG", period=period)
    lqd_s = _download_close_series("LQD", period=period)
    uup_s = _download_close_series("UUP", period=period)
    oil_s = _download_close_series("CL=F", period=period)
    spy_s = _download_close_series("SPY", period=period)
    tlt_s = _download_close_series("TLT", period=period)

    if len(hyg_s) < 22 or len(lqd_s) < 22:
        raise RuntimeError("Macro data requires at least 22 observations for HYG and LQD.")
    if len(uup_s) < 22:
        raise RuntimeError("Macro data requires at least 22 observations for UUP.")
    if len(oil_s) < 22:
        raise RuntimeError("Macro data requires at least 22 observations for CL=F.")
    if len(spy_s) < 6 or len(tlt_s) < 6:
        raise RuntimeError("Macro data requires at least 6 observations for SPY and TLT.")

    values = {
        "vix": float(vix_s.iloc[-1]),
        "yield_10y": float(tnx_s.iloc[-1]),
        "yield_3m": float(irx_s.iloc[-1]),
    }
    values["yield_spread"] = values["yield_10y"] - values["yield_3m"]
    values["credit_spread"] = float((hyg_s.iloc[-1] / hyg_s.iloc[-22] - 1) - (lqd_s.iloc[-1] / lqd_s.iloc[-22] - 1))
    values["usd_strength"] = float(uup_s.iloc[-1] / uup_s.iloc[-22] - 1)
    values["oil_mom_1m"] = float(oil_s.iloc[-1] / oil_s.iloc[-22] - 1)
    ratio_now = float(spy_s.iloc[-1] / tlt_s.iloc[-1])
    ratio_prev = float(spy_s.iloc[-6] / tlt_s.iloc[-6])
    if ratio_prev == 0:
        raise RuntimeError("Macro data SPY/TLT ratio cannot use zero previous ratio.")
    values["spy_tlt_ratio"] = float(ratio_now / ratio_prev - 1)

    missing = [field for field in REQUIRED_MACRO_FIELDS if field not in values]
    if missing:
        raise RuntimeError(f"Macro snapshot missing required fields: {missing}")

    logger.info(
        "Macro snapshot loaded: VIX=%.1f Spread=%.2f Credit=%+.3f USD=%+.3f",
        values["vix"],
        values["yield_spread"],
        values["credit_spread"],
        values["usd_strength"],
    )
    return MacroData(values=values, source="yfinance.macro_snapshot")
