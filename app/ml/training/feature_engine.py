import os
import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import pandas_ta as ta

# Minimum trading bars needed to populate all rolling windows:
#   mom_6m  = 126 bars
#   vol_60d = 60 bars
#   MACD slow=26 + signal=9 ≈ 35 bars
# We round up to 200 so the whole feature row is always filled.
REQUIRED_BARS = 200

# Macro tickers we download alongside asset prices
MACRO_TICKERS = {
    "^VIX":  "vix",        # Implied-volatility / fear gauge
    "^TNX":  "yield_10y",  # US 10-year Treasury yield
    "^IRX":  "yield_3m",   # US 3-month T-bill (≈ short end of curve)
    "HYG":   "hyg",        # High-Yield Bond ETF (credit proxy)
    "LQD":   "lqd",        # Investment-Grade Bond ETF
    "UUP":   "uup",        # US Dollar Index ETF
    "CL=F":  "oil",        # WTI Crude Oil Futures
}


def build_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame whose columns are the raw macro prices/levels
    (indexed by date), compute derived macro signals and return a
    daily macro-feature DataFrame aligned to the same index.

    Derived columns:
        vix             – VIX level (already a signal, no transform needed)
        yield_10y       – 10Y Treasury yield level
        yield_3m        – 3M T-bill level
        yield_spread    – 10Y minus 3M (negative → inverted curve → recession risk)
        credit_spread   – 21-day return HYG minus 21-day return LQD (risk appetite)
        usd_strength    – 21-day return of UUP (dollar momentum)
        oil_mom_1m      – 21-day return of WTI crude (inflation proxy)
        spy_tlt_ratio   – Daily close SPY/TLT (risk-on vs risk-off ratio)
    """
    m = pd.DataFrame(index=macro_df.index)

    def _col(name: str) -> pd.Series:
        return macro_df[name] if name in macro_df.columns else pd.Series(dtype=float)

    # Level-based features -------------------------------------------------
    m["vix"]          = _col("vix")
    m["yield_10y"]    = _col("yield_10y")
    m["yield_3m"]     = _col("yield_3m")
    m["yield_spread"] = _col("yield_10y") - _col("yield_3m")

    # Return-based spreads ------------------------------------------------
    hyg = _col("hyg")
    lqd = _col("lqd")
    if not hyg.empty and not lqd.empty:
        m["credit_spread"] = hyg.pct_change(21) - lqd.pct_change(21)
    else:
        m["credit_spread"] = 0.0

    m["usd_strength"] = _col("uup").pct_change(21) if not _col("uup").empty else 0.0
    m["oil_mom_1m"]   = _col("oil").pct_change(21) if not _col("oil").empty else 0.0

    # Risk-on ratio --------------------------------------------------------
    # We need SPY and TLT — they may appear in macro_df if the training script
    # adds them; otherwise default to 0.
    if "spy" in macro_df.columns and "tlt" in macro_df.columns:
        ratio = macro_df["spy"] / macro_df["tlt"]
        m["spy_tlt_ratio"] = ratio.pct_change(5)   # 1-week change in the ratio
    else:
        m["spy_tlt_ratio"] = 0.0

    return m.ffill().fillna(0)


def calculate_features(
    price_series: pd.Series,
    macro_row: pd.Series = None,
) -> pd.DataFrame:
    """
    Calculates technical + macro features for a single asset.

    Asset-specific features (computed from price_series):
        mom_1m, mom_3m, mom_6m       – price momentum
        vol_20d, vol_60d             – realised volatility (annualised)
        rsi_14                       – RSI
        macd_hist                    – MACD histogram
        bb_pct                       – Bollinger %B (position within band)

    Macro features (broadcast from macro_row, or defaults if None):
        vix, yield_10y, yield_3m, yield_spread,
        credit_spread, usd_strength, oil_mom_1m, spy_tlt_ratio

    Returns DataFrame with all 16 features (no 'close' column).
    """
    df = pd.DataFrame({"close": price_series})

    # ── Asset-level technical features ────────────────────────────────────
    df["mom_1m"] = df["close"].pct_change(21)
    df["mom_3m"] = df["close"].pct_change(63)
    df["mom_6m"] = df["close"].pct_change(126)

    df["vol_20d"] = df["close"].pct_change().rolling(20).std() * (252 ** 0.5)
    df["vol_60d"] = df["close"].pct_change().rolling(60).std() * (252 ** 0.5)

    df["rsi_14"] = df.ta.rsi(length=14)

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df["macd_hist"] = (
        macd["MACDh_12_26_9"]
        if (macd is not None and "MACDh_12_26_9" in macd.columns)
        else 0.0
    )

    bbands = df.ta.bbands(length=20, std=2)
    df["bb_pct"] = (
        bbands["BBP_20_2.0"]
        if (bbands is not None and "BBP_20_2.0" in bbands.columns)
        else 0.5
    )

    # ── Macro features (scalar broadcast) ────────────────────────────────
    defaults = {
        "vix":           20.0,
        "yield_10y":      4.5,
        "yield_3m":       5.0,
        "yield_spread":  -0.5,
        "credit_spread":  0.0,
        "usd_strength":   0.0,
        "oil_mom_1m":     0.0,
        "spy_tlt_ratio":  0.0,
    }
    for col, default in defaults.items():
        df[col] = float(macro_row.get(col, default)) if macro_row is not None else default

    df = df.drop(columns=["close"])
    return df


# Canonical feature column order (must match training exactly)
FEATURE_COLUMNS = [
    "mom_1m", "mom_3m", "mom_6m",
    "vol_20d", "vol_60d",
    "rsi_14", "macd_hist", "bb_pct",
    "vix", "yield_10y", "yield_3m", "yield_spread",
    "credit_spread", "usd_strength", "oil_mom_1m", "spy_tlt_ratio",
]
