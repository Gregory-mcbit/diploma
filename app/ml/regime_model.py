from __future__ import annotations

from typing import Mapping

from app.domain.schemas import RegimeModelOutput, RegimeType


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def infer_regime(
    *,
    macro_data: Mapping[str, float],
    spy_features: Mapping[str, float],
) -> RegimeModelOutput:
    vix = float(macro_data["vix"])
    spread = float(macro_data["yield_spread"])
    credit_spread = float(macro_data["credit_spread"])
    spy_tlt_ratio = float(macro_data["spy_tlt_ratio"])
    usd_strength = float(macro_data["usd_strength"])
    oil_mom_1m = float(macro_data["oil_mom_1m"])
    spy_mom_1m = float(spy_features["mom_1m"])
    spy_mom_3m = float(spy_features.get("mom_3m", spy_mom_1m))

    signal_components = {
        "vix": vix,
        "yield_spread": spread,
        "credit_spread": credit_spread,
        "spy_tlt_ratio": spy_tlt_ratio,
        "usd_strength": usd_strength,
        "oil_mom_1m": oil_mom_1m,
        "spy_mom_1m": spy_mom_1m,
        "spy_mom_3m": spy_mom_3m,
    }

    stress_score = 0.0
    stress_score += 2.0 if vix >= 30 else 1.0 if vix >= 22 else -1.0
    stress_score += 2.0 if spread < 0 else 0.0
    stress_score += 1.0 if credit_spread < 0 else -0.5
    stress_score += 1.0 if spy_tlt_ratio < 0 else -0.5
    stress_score += 1.0 if spy_mom_1m < 0 else -0.5
    stress_score += 0.5 if spy_mom_3m < 0 else -0.25
    stress_score += 0.5 if usd_strength > 0.03 else 0.0

    uncertainty_notes: list[str] = []
    if abs(spread) < 0.15:
        uncertainty_notes.append("Yield curve signal is close to neutral.")
    if abs(spy_mom_1m) < 0.01:
        uncertainty_notes.append("Short-term equity momentum is weak and directionally ambiguous.")
    if abs(credit_spread) < 0.01:
        uncertainty_notes.append("Credit spread signal is mild and adds limited conviction.")

    if vix >= 30 and stress_score >= 4.0:
        regime = RegimeType.high_volatility
    elif stress_score >= 3.0:
        regime = RegimeType.risk_off
    elif spy_mom_1m < 0 and spread < 0:
        regime = RegimeType.bearish
    elif vix < 18 and spread > 0 and spy_mom_1m > 0.02 and spy_tlt_ratio > 0:
        regime = RegimeType.bullish
    elif stress_score <= -1.0 and spy_mom_3m > 0:
        regime = RegimeType.risk_on
    else:
        regime = RegimeType.sideways

    confidence = _clamp(0.55 + min(abs(stress_score), 4.0) * 0.08, 0.55, 0.92)
    is_risk_off = regime in {RegimeType.bearish, RegimeType.high_volatility, RegimeType.risk_off}

    return RegimeModelOutput(
        regime=regime,
        confidence=round(confidence, 2),
        is_risk_off=is_risk_off,
        signal_components=signal_components,
        uncertainty_notes=uncertainty_notes,
    )
