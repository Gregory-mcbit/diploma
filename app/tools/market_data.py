import os
import uuid
import pandas as pd
import yfinance as yf
from typing import List
from app.observability.logger import get_logger


CACHE_DIR = "data/cache/"
os.makedirs(CACHE_DIR, exist_ok=True)
logger = get_logger(__name__)


def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> str:
    """
    Fetches raw market data (Adj Close) using yfinance.
    Saves it to a parquet file and returns the UUID filepath.
    """
    logger.info("Fetching market data for %s tickers from %s to %s.", len(tickers), start_date, end_date)
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError("Market data download returned no rows.")
    
    # Extract only Adjusted Close prices
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise RuntimeError("Market data response is missing 'Adj Close' prices.")
        prices = data["Adj Close"]
    else:
        if "Adj Close" not in data.columns:
            raise RuntimeError("Market data response is missing 'Adj Close' prices.")
        prices = pd.DataFrame(data["Adj Close"])
        prices.columns = tickers

    # Handle missing data forward-filling and backward-filling
    prices = prices.ffill().bfill()
    if prices.empty:
        raise RuntimeError("Market price matrix is empty after cleaning.")
    
    # Cache to disk
    file_id = str(uuid.uuid4())
    filepath = os.path.join(CACHE_DIR, f"{file_id}_prices.parquet")
    prices.to_parquet(filepath)
    logger.info("Cached market data to %s.", filepath)
    
    return filepath


def load_market_data(filepath: str) -> pd.DataFrame:
    """Loads a previously cached parquet price matrix."""
    return pd.read_parquet(filepath)
