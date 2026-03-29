import os
import uuid
import pandas as pd
import yfinance as yf
from typing import List


CACHE_DIR = "data/cache/"
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> str:
    """
    Fetches raw market data (Adj Close) using yfinance.
    Saves it to a parquet file and returns the UUID filepath.
    """
    print(f"[TOOL] Fetching real market data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    
    # Extract only Adjusted Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = pd.DataFrame(data["Adj Close"])
        prices.columns = tickers

    # Handle missing data forward-filling and backward-filling
    prices = prices.ffill().bfill()
    
    # Cache to disk
    file_id = str(uuid.uuid4())
    filepath = os.path.join(CACHE_DIR, f"{file_id}_prices.parquet")
    prices.to_parquet(filepath)
    
    return filepath


def load_market_data(filepath: str) -> pd.DataFrame:
    """Loads a previously cached parquet price matrix."""
    return pd.read_parquet(filepath)
