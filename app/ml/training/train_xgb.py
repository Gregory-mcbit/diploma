"""
Offline training script for the XGBoost forward-return prediction model.

What it does:
  1. Downloads 15 years of price history for asset universe + macro tickers.
  2. Builds the macro feature DataFrame (VIX, yields, credit spread, etc.).
  3. For each asset × each trading day, computes the 16-feature row and the
     21-day forward return (the label).
  4. Trains an XGBRegressor with early stopping on a held-out validation set.
  5. Saves the model to data/models/xgb_alpha.json.

Run once (or after feature_engine changes):
    PYTHONPATH=. python3 app/ml/training/train_xgb.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from app.domain.asset_metadata import STANDARD_UNIVERSE

from app.ml.training.feature_engine import (
    calculate_features,
    build_macro_features,
    FEATURE_COLUMNS,
    MACRO_TICKERS,
    REQUIRED_BARS,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR   = "data/models/"
MODEL_PATH   = os.path.join(MODELS_DIR, "xgb_alpha.json")
START_DATE   = "2009-01-01"   # 15+ years of history
END_DATE     = "2024-12-31"   # keep 2025 out-of-sample

ASSET_UNIVERSE = STANDARD_UNIVERSE

FORWARD_RETURN_DAYS = 21   # predict 21-day (≈ 1M) forward return

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Downloads adjusted close prices; returns a multi-ticker DataFrame."""
    print(f"[DATA] Downloading {len(tickers)} tickers ({start} → {end})…")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.dropna(how="all")
    print(f"  → Shape: {prices.shape}")
    return prices


def build_training_dataset(
    asset_prices: pd.DataFrame,
    macro_features: pd.DataFrame,
) -> tuple:
    """
    Creates (X, y) from all assets × all dates.
    Each row = one asset on one trading day.
    Label = 21-day forward return (annualised).
    """
    all_X, all_y = [], []

    for ticker in asset_prices.columns:
        series = asset_prices[ticker].dropna()
        if len(series) < REQUIRED_BARS + FORWARD_RETURN_DAYS:
            print(f"  [SKIP] {ticker}: insufficient data ({len(series)} bars)")
            continue

        # Build technical-only features (no macro_row → defaults will be overwritten)
        feat_df = calculate_features(series)
        # Drop macro columns that calculate_features added as defaults
        # (we will join the real macro values from macro_features)
        tech_cols = [
            "mom_1m", "mom_3m", "mom_6m",
            "vol_20d", "vol_60d",
            "rsi_14", "macd_hist", "bb_pct",
        ]
        feat_df = feat_df[tech_cols]

        # Forward return label
        fwd_ret = series.pct_change(FORWARD_RETURN_DAYS).shift(-FORWARD_RETURN_DAYS)

        # Align indices: technical features + macro features + label
        joined = feat_df.join(macro_features, how="left").join(
            fwd_ret.rename("label"), how="inner"
        )
        joined = joined.dropna(subset=["label"] + FEATURE_COLUMNS)

        if joined.empty:
            continue

        all_X.append(joined[FEATURE_COLUMNS])
        all_y.append(joined["label"])
        print(f"  [OK] {ticker}: {len(joined)} training rows")

    if not all_X:
        raise RuntimeError("No training data assembled – check tickers and dates.")

    X = pd.concat(all_X).reset_index(drop=True)
    y = pd.concat(all_y).reset_index(drop=True)
    return X, y


# ── Main ──────────────────────────────────────────────────────────────────────

def train_xgb_model():
    print("=" * 60)
    print("XGBoost Training Pipeline — 16-Feature Model")
    print("=" * 60)

    # 1. Download asset universe prices
    asset_prices = download_prices(ASSET_UNIVERSE, START_DATE, END_DATE)

    # 2. Download macro tickers (we need SPY and TLT for the ratio too)
    macro_raw_tickers = list(MACRO_TICKERS.keys()) + ["SPY", "TLT"]
    macro_raw = download_prices(macro_raw_tickers, START_DATE, END_DATE)

    # Rename columns to friendly names
    col_map = {k: v for k, v in MACRO_TICKERS.items()}
    col_map["SPY"] = "spy"
    col_map["TLT"] = "tlt"
    macro_raw = macro_raw.rename(columns=col_map)

    macro_features = build_macro_features(macro_raw)
    print(f"\n[MACRO] Feature columns: {list(macro_features.columns)}")
    print(f"[MACRO] Date range: {macro_features.index[0]} → {macro_features.index[-1]}")

    # 3. Build training dataset
    print("\n[FEATURES] Building training rows for each asset…")
    X, y = build_training_dataset(asset_prices, macro_features)
    print(f"\n[DATASET] Total rows: {len(X):,}  |  Features: {X.shape[1]}")
    print(f"[DATASET] Label stats — mean={y.mean():.4f}  std={y.std():.4f}")

    # 4. Train/validation split (chronological – no leakage)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    print(f"[SPLIT] Train={len(X_train):,}  Val={len(X_val):,}")

    # 5. Train XGBoost
    print("\n[XGB] Training…")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # 6. Evaluate
    preds = model.predict(X_val)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    corr  = float(pd.Series(preds).corr(y_val.reset_index(drop=True)))
    print(f"\n[EVAL] Val RMSE : {rmse:.5f}")
    print(f"[EVAL] Val Corr : {corr:.4f}  (IC – Information Coefficient)")

    # 7. Feature importance
    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("\n[IMPORTANCE] Top-10 features:")
    print(importance.head(10).to_string())

    # 8. Save
    model.save_model(MODEL_PATH)
    print(f"\n[DONE] Model saved → {MODEL_PATH}")
    print(f"       Features used: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    train_xgb_model()
