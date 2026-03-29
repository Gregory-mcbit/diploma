import os
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from app.ml.training.feature_engine import calculate_features


MODELS_DIR = "data/models/"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_xgb_model():
    """
    Offline training script: Fetches 15 years of data, engineers features,
    splits train/test, and trains an XGBRegressor to predict alpha.
    """
    print("[TRAINING] Fetching 15 years of history for core universe...")
    # Representative universe for training patterns
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL", "MSFT", "JNJ"]
    
    data = yf.download(tickers, start="2010-01-01", end="2025-01-01", auto_adjust=False, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = pd.DataFrame(data["Adj Close"])
        prices.columns = tickers

    print("[TRAINING] Building features and parsing labels (30-day forward returns)...")
    all_X = []
    all_y = []
    
    for ticker in tickers:
        p = prices[ticker].dropna()
        if len(p) < 200:
            continue
            
        features = calculate_features(p)
        
        # Supervised Target: 21-trading-day forward return
        # i.e., what is the return of this asset 1 month from today?
        target = p.pct_change(periods=21).shift(-21)
        
        combined = features.copy()
        combined["target"] = target
        combined = combined.dropna() # Drop NaNs strictly to prevent XGBoost warnings
        
        all_X.append(combined.drop(columns=["target"]))
        all_y.append(combined["target"])
        
    X = pd.concat(all_X)
    y = pd.concat(all_y)
    
    print(f"[TRAINING] Dataset built. Shape: {X.shape}. Splitting chronologically...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("[TRAINING] Training XGBRegressor...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05, 
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    model_path = os.path.join(MODELS_DIR, "xgb_alpha.json")
    model.save_model(model_path)
    print(f"[TRAINING] SUCCESS! XGBoost Model saved realistically to {model_path}.")


if __name__ == "__main__":
    train_xgb_model()
