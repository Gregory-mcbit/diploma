import pandas as pd
import pandas_ta as ta


def calculate_features(price_series: pd.Series) -> pd.DataFrame:
    """
    Calculates technical analysis features for a single asset's price series.
    Returns a DataFrame of features alongside the original series (optional).
    """
    df = pd.DataFrame({"close": price_series})
    
    # Momentum (Rolling Returns approximating 1M, 3M, 6M)
    df["mom_1m"] = df["close"].pct_change(periods=21)
    df["mom_3m"] = df["close"].pct_change(periods=63)
    df["mom_6m"] = df["close"].pct_change(periods=126)
    
    # Volatility (Annualized rolling standard deviation)
    df["vol_20d"] = df["close"].pct_change().rolling(20).std() * (252**0.5)
    df["vol_60d"] = df["close"].pct_change().rolling(60).std() * (252**0.5)
    
    # Technicals via pandas_ta for quality/trend proxies
    df["rsi_14"] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None and "MACDh_12_26_9" in macd.columns:
        df["macd_hist"] = macd["MACDh_12_26_9"]
    else:
        df["macd_hist"] = 0.0 # fallback
        
    # Drop original close price so it's only features
    return df.drop(columns=["close"])
