import os
import xgboost as xgb
import pandas as pd
from typing import List, Dict
from app.domain.schemas import AssetScore, FactorScores
from app.ml.training.feature_engine import calculate_features


MODEL_PATH = "data/models/xgb_alpha.json"


def run_ml_scoring_pipeline(universe: List[str], price_df: pd.DataFrame) -> Dict[str, AssetScore]:
    """
    Real online inference pipeline. Loads the pre-trained XGBoost model 
    and predicts alpha for current assets without recalculating/retraining.
    """
    results = {}
    naive = False
    
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] XGBoost model ({MODEL_PATH}) not found! Returning naive scores. Did you run train_xgb.py?")
        naive = True
    else:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)

    for ticker in universe:
        if ticker not in price_df.columns:
            continue
        
        p = price_df[ticker].dropna()
        if len(p) < 150: # Not enough history to fill rolling 126d features
            continue
            
        # Get strictly the latest row of features for inference
        features = calculate_features(p).iloc[-1:]
        features = features.fillna(0) # clean rare NaNs
        
        if naive:
            # Fallback naive logic
            alpha_pred = float(features["mom_1m"].values[0])
        else:
            # Real XGBoost Inference Call
            alpha_pred = float(model.predict(features)[0])
            
        factors = FactorScores(
            momentum=float(features["mom_3m"].values[0]),
            volatility=float(features["vol_20d"].values[0]),
            quality=float(features["macd_hist"].values[0]),
            extra_factors={
                "rsi": float(features["rsi_14"].values[0]),
                "xgb_alpha": alpha_pred
            }
        )
        
        results[ticker] = AssetScore(
            asset_ticker=ticker,
            factors=factors,
            overall_score=alpha_pred,
            confidence=0.85 # XGBoost predict margins can be mapped here later
        )
        
    return results
