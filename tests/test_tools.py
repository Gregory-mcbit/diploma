import pytest
import pandas as pd
import numpy as np
from app.tools.optimizer import optimize_portfolio
from app.tools.risk_metrics import generate_risk_report
from app.domain.schemas import AssetScore, Constraints, EffectivePolicy, FactorScores

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


def test_optimizer_respects_cash_sleeve_feasibility():
    dates = pd.date_range("2023-01-01", periods=120)
    np.random.seed(7)
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.0012, 0.018, 120),
            "MSFT": np.random.normal(0.0010, 0.017, 120),
            "JNJ": np.random.normal(0.0007, 0.012, 120),
        },
        index=dates,
    )
    prices = 100 * np.cumprod(1 + returns)
    scores = {
        "AAPL": AssetScore(
            asset_ticker="AAPL",
            factors=FactorScores(momentum=0.2, volatility=0.3, quality=0.5),
            overall_score=0.07,
            confidence=0.9,
        ),
        "MSFT": AssetScore(
            asset_ticker="MSFT",
            factors=FactorScores(momentum=0.15, volatility=0.25, quality=0.5),
            overall_score=0.05,
            confidence=0.85,
        ),
        "JNJ": AssetScore(
            asset_ticker="JNJ",
            factors=FactorScores(momentum=0.05, volatility=0.12, quality=0.7),
            overall_score=0.03,
            confidence=0.8,
        ),
    }
    constraints = Constraints(
        max_asset_weight=0.30,
        max_sector_weight=1.0,
        allowed_asset_classes=["stocks"],
        forbidden_assets=[],
        max_drawdown_tolerance=0.2,
        min_cash_weight=0.10,
    )

    weights = optimize_portfolio(prices, scores, constraints)

    assert abs(sum(weights.values()) - 0.90) < 1e-3
    assert all(weight <= constraints.max_asset_weight + 1e-4 for weight in weights.values())

def test_risk_violations(dummy_data):
    prices, _, _ = dummy_data
    # Artificially inject a violation where AAPL is 80%, but limit is 50%
    weights = {"AAPL": 0.8, "MSFT": 0.2} 
    report = generate_risk_report(weights, prices, max_asset_weight=0.5)

    assert len(report.violations) > 0
    assert "AAPL" in report.violations[0]
    assert report.fit_to_profile == "нарушение"
    
def test_risk_volatility(dummy_data):
    prices, _, _ = dummy_data
    weights = {"AAPL": 0.5, "MSFT": 0.5} 
    report = generate_risk_report(weights, prices, max_asset_weight=0.6)
    
    # Ensure math is running
    assert report.portfolio_volatility > 0.0
    assert report.max_drawdown_estimate >= 0.0
    assert report.concentration_hhi == pytest.approx(0.5)
    assert report.var_95 >= 0.0
    assert -1.0 <= report.avg_correlation <= 1.0
    assert report.fit_to_profile == "допустимо"


def test_drawdown_breach_is_warning_not_hard_violation():
    dates = pd.date_range("2023-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "AAPL": [100, 102, 101, 90, 88, 92, 95, 97],
            "MSFT": [100, 101, 100, 91, 89, 91, 94, 96],
        },
        index=dates,
    )
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    constraints = Constraints(
        max_asset_weight=0.8,
        max_sector_weight=1.0,
        allowed_asset_classes=["stocks"],
        forbidden_assets=[],
        max_drawdown_tolerance=0.05,
    )

    report = generate_risk_report(weights, prices, constraints=constraints)

    assert not any("допустимой просадки" in violation.lower() for violation in report.violations)
    assert any("просадка выше целевого лимита профиля" in warning.lower() for warning in report.warnings)


def test_risk_report_flags_restricted_policy_exposures(dummy_data):
    prices, _, constraints = dummy_data
    weights = {"AAPL": 0.4, "GLD": 0.4}
    extended_prices = prices.copy()
    extended_prices["GLD"] = np.linspace(100.0, 110.0, len(extended_prices))

    report = generate_risk_report(
        weights,
        extended_prices,
        constraints=constraints,
        effective_policy=EffectivePolicy(
            constraints=constraints,
            restricted_sectors=["technology"],
            restricted_countries=[],
            income_preference=None,
        ),
    )

    assert any("нарушение ограничения по сектору" in violation.lower() for violation in report.violations)
    assert report.fit_to_profile == "нарушение"


def test_risk_report_flags_income_preference_breaches(dummy_data):
    prices, _, constraints = dummy_data
    weights = {"AAPL": 0.4, "MSFT": 0.4}

    report = generate_risk_report(
        weights,
        prices,
        constraints=constraints,
        effective_policy=EffectivePolicy(
            constraints=constraints,
            income_preference="income",
            min_income_weight=0.35,
            max_growth_weight=0.30,
        ),
    )

    assert any("нарушение income preference" in violation.lower() for violation in report.violations)
    assert report.fit_to_profile == "нарушение"
