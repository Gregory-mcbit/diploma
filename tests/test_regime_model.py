from app.domain.schemas import RegimeType
from app.ml.regime_model import infer_regime


def test_infer_regime_identifies_risk_off_conditions():
    result = infer_regime(
        macro_data={
            "vix": 31.0,
            "yield_spread": -0.4,
            "credit_spread": -0.03,
            "spy_tlt_ratio": -0.02,
            "usd_strength": 0.04,
            "oil_mom_1m": -0.01,
        },
        spy_features={
            "mom_1m": -0.03,
            "mom_3m": -0.05,
        },
    )

    assert result.regime in {RegimeType.risk_off, RegimeType.high_volatility}
    assert result.is_risk_off is True
    assert result.confidence >= 0.7
    assert "vix" in result.signal_components


def test_infer_regime_identifies_bullish_conditions():
    result = infer_regime(
        macro_data={
            "vix": 15.0,
            "yield_spread": 0.8,
            "credit_spread": 0.02,
            "spy_tlt_ratio": 0.04,
            "usd_strength": 0.0,
            "oil_mom_1m": 0.03,
        },
        spy_features={
            "mom_1m": 0.04,
            "mom_3m": 0.08,
        },
    )

    assert result.regime in {RegimeType.bullish, RegimeType.risk_on}
    assert result.confidence >= 0.6

