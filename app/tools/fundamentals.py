from __future__ import annotations

from typing import Dict, List

import yfinance as yf

from app.domain.schemas import FundamentalSnapshot
from app.observability.logger import get_logger


logger = get_logger(__name__)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_fundamental_metrics(info: dict) -> dict[str, float | None]:
    metrics = {
        "pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "roe": _safe_float(info.get("returnOnEquity")),
        "net_margin": _safe_float(info.get("profitMargins")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
    }
    return metrics


def fetch_fundamentals(tickers: List[str]) -> Dict[str, FundamentalSnapshot]:
    snapshots: Dict[str, FundamentalSnapshot] = {}
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        if not isinstance(info, dict) or not info:
            raise RuntimeError(f"Fundamental data unavailable for {ticker}.")
        metrics = _normalize_fundamental_metrics(info)
        snapshots[ticker] = FundamentalSnapshot(
            ticker=ticker,
            metrics=metrics,
            source="yfinance.info",
        )
    logger.info("Loaded fundamentals for %s tickers.", len(snapshots))
    return snapshots
