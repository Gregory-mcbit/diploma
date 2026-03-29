import pytest
import pandas as pd
import numpy as np
from app.tools.optimizer import optimize_portfolio
from app.tools.risk_metrics import generate_risk_report
from app.domain.schemas import AssetScore, FactorScores, Constraints

@pytest.fixture
def dummy_data():
    dates = pd.date_range("2023-01-01", periods=100)
    np.random.seed(42)
    # Create two artificial price series (one trending up, one flat)
    df = pd.DataFrame({
        "AAPL": np.random.normal(0.001, 0.02, 100),
        "MSFT": np.random.normal(0.000, 0.015, 100)
    }, index=dates)
    prices = 100 * np.cumprod(1 + df)
    
    scores = {
        "AAPL": AssetScore(
            asset_ticker="AAPL", 
            factors=FactorScores(momentum=0.1, volatility=0.2, quality=0.5), 
            overall_score=0.08, # High alpha
            confidence=0.9
        ),
        "MSFT": AssetScore(
            asset_ticker="MSFT", 
            factors=FactorScores(momentum=0.0, volatility=0.1, quality=0.5), 
            overall_score=0.01, # Low alpha
            confidence=0.8
        )
    }
    
    constraints = Constraints(
        max_asset_weight=0.6, # Max 60% per asset
        max_sector_weight=1.0,
        allowed_asset_classes=["stocks"],
        forbidden_assets=[],
        max_drawdown_tolerance=0.2
    )
    return prices, scores, constraints

def test_optimizer_weights(dummy_data):
    prices, scores, constraints = dummy_data
    weights = optimize_portfolio(prices, scores, constraints)
    
    # 1. Weights must sum to near 1.0 (some rounding/cash handling may leave miniscule float residuals)
    assert abs(sum(weights.values()) - 1.0) < 1e-3, f"Weights sum to {sum(weights.values())}"
    
    # 2. AAPL has a much higher ML score. It should ideally hit the max cap (0.6).
    assert "AAPL" in weights
    assert weights["AAPL"] <= constraints.max_asset_weight + 1e-4
    assert weights["AAPL"] >= 0.50 # Given the massive divergence in alpha, it should be heavily allocated

def test_risk_violations(dummy_data):
    prices, _, _ = dummy_data
    # Artificially inject a violation where AAPL is 80%, but limit is 50%
    weights = {"AAPL": 0.8, "MSFT": 0.2} 
    report = generate_risk_report(weights, prices, max_asset_weight=0.5)
    
    assert len(report.violations) > 0
    assert "AAPL" in report.violations[0]
    assert report.fit_to_profile == "violation"
    
def test_risk_volatility(dummy_data):
    prices, _, _ = dummy_data
    weights = {"AAPL": 0.5, "MSFT": 0.5} 
    report = generate_risk_report(weights, prices, max_asset_weight=0.6)
    
    # Ensure math is running
    assert report.portfolio_volatility > 0.0
    assert report.max_drawdown_estimate >= 0.0
    assert report.fit_to_profile == "acceptable"
