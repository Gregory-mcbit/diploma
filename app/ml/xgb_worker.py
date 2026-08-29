#!/usr/bin/env python3
"""
Isolated XGBoost inference worker.
Spawned as a subprocess to avoid OpenMP/libomp segfault on macOS ARM.
Reads a JSON payload from stdin, writes JSON scores to stdout.
"""
import os
import sys
import json

os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import xgboost as xgb

MODEL_PATH = "data/models/xgb_alpha.json"
sys.path.insert(0, ".")


def main():
    payload = json.loads(sys.stdin.read())
    parquet_path: str = payload["parquet_path"]
    universe: list   = payload["universe"]
    macro: dict      = payload.get("macro", {})

    from app.ml.training.feature_engine import (
        calculate_features,
        FEATURE_COLUMNS,
        REQUIRED_BARS,
    )

    price_df  = pd.read_parquet(parquet_path)
    macro_row = pd.Series(macro)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = xgb.XGBRegressor(n_jobs=1)
    model.load_model(MODEL_PATH)

    results = {}

    for ticker in universe:
        if ticker not in price_df.columns:
            raise RuntimeError(f"[WORKER] {ticker}: missing from parquet columns.")

        series = price_df[ticker].dropna()
        if len(series) < REQUIRED_BARS:
            raise RuntimeError(f"[WORKER] {ticker}: only {len(series)} bars, need {REQUIRED_BARS}.")

        features_df = calculate_features(series, macro_row=macro_row)
        latest = features_df.iloc[-1:]

        # Ensure columns match training order exactly
        missing = [c for c in FEATURE_COLUMNS if c not in latest.columns]
        if missing:
            raise RuntimeError(f"[WORKER] {ticker}: missing required features {missing}.")
        latest = latest[FEATURE_COLUMNS]
        if latest.isna().any().any():
            null_cols = [col for col in latest.columns if latest[col].isna().any()]
            raise RuntimeError(f"[WORKER] {ticker}: NaN values detected in required features {null_cols}.")

        try:
            alpha = float(model.predict(latest)[0])
        except ValueError as e:
            raise RuntimeError(f"[WORKER] {ticker}: predict failed ({e}).") from e

        results[ticker] = alpha

    print(json.dumps(results))


if __name__ == "__main__":
    main()
