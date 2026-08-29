from app.agents.portfolio import _apply_income_preference_policy, _filter_universe_by_policy
from app.domain.schemas import AssetScore, Constraints, FactorScores, RegimeReport, RegimeType, RiskProfile, InvestorProfile
from app.policy_engine import derive_effective_policy
from app.rag.structured_policy_rules import load_policy_rules


def test_policy_rules_loader_validates_external_config():
    rules = load_policy_rules()

    assert len(rules) >= 5
    assert all(rule.id for rule in rules)
    assert any(rule.id == "risk-off-defensive-cap" for rule in rules)


def test_policy_engine_derives_structured_risk_off_policy():
    profile = InvestorProfile(
        risk_profile=RiskProfile.moderate,
        horizon_years=10,
        target="Balanced growth",
        constraints=Constraints(
            max_asset_weight=0.25,
            max_sector_weight=0.40,
            allowed_asset_classes=["stocks", "bonds", "commodities"],
            forbidden_assets=[],
            max_drawdown_tolerance=0.25,
            min_cash_weight=0.0,
            max_correlation_threshold=0.95,
        ),
    )
    regime = RegimeReport(
        current_regime=RegimeType.risk_off,
        confidence=0.80,
        drivers=["Elevated volatility"],
        is_risk_off=True,
    )

    effective_policy = derive_effective_policy(profile, regime)

    assert effective_policy.constraints.max_asset_weight == 0.10
    assert effective_policy.constraints.min_cash_weight == 0.10
    assert effective_policy.constraints.max_sector_weight == 0.35
    assert effective_policy.constraints.max_drawdown_tolerance == 0.20
    assert effective_policy.constraints.max_correlation_threshold == 0.85
    assert "risk-off-defensive-cap" in effective_policy.applied_rule_ids
    assert "global-sector-cap" in effective_policy.applied_rule_ids


def test_policy_engine_applies_large_cap_and_income_preferences():
    profile = InvestorProfile(
        risk_profile=RiskProfile.moderate,
        horizon_years=10,
        target="Income-oriented diversified growth",
        investment_amount=2_000_000,
        income_preference="income",
        sector_restrictions=["technology"],
        country_restrictions=["us"],
        constraints=Constraints(
            max_asset_weight=0.25,
            max_sector_weight=0.40,
            allowed_asset_classes=["stocks", "bonds", "commodities"],
            forbidden_assets=[],
            max_drawdown_tolerance=0.25,
            min_cash_weight=0.0,
            max_correlation_threshold=0.95,
        ),
    )
    regime = RegimeReport(
        current_regime=RegimeType.sideways,
        confidence=0.75,
        drivers=["Neutral macro"],
        is_risk_off=False,
    )

    effective_policy = derive_effective_policy(profile, regime)

    assert effective_policy.constraints.max_asset_weight == 0.15
    assert effective_policy.constraints.max_sector_weight == 0.30
    assert effective_policy.constraints.min_cash_weight == 0.03
    assert effective_policy.restricted_sectors == ["technology"]
    assert effective_policy.restricted_countries == ["us"]
    assert effective_policy.income_preference == "income"
    assert effective_policy.min_income_weight == 0.35
    assert effective_policy.max_growth_weight == 0.30
    assert "profile-large-cap-diversification" in effective_policy.applied_rule_ids
    assert "profile-income-cash-buffer" in effective_policy.applied_rule_ids
    assert "profile-sector-restrictions" in effective_policy.applied_rule_ids
    assert "profile-country-restrictions" in effective_policy.applied_rule_ids


def test_portfolio_policy_filter_uses_effective_policy_restrictions():
    effective_policy = derive_effective_policy(
        InvestorProfile(
            risk_profile=RiskProfile.moderate,
            horizon_years=10,
            target="Defensive income",
            income_preference="income",
            sector_restrictions=["technology"],
            country_restrictions=["us"],
            constraints=Constraints(
                max_asset_weight=0.25,
                max_sector_weight=0.40,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.25,
                min_cash_weight=0.0,
                max_correlation_threshold=0.95,
            ),
        ),
        RegimeReport(
            current_regime=RegimeType.sideways,
            confidence=0.70,
            drivers=["Neutral macro"],
            is_risk_off=False,
        ),
    )
    asset_scores = {
        "QQQ": AssetScore(
            asset_ticker="QQQ",
            factors=FactorScores(momentum=0.3, volatility=-0.2, quality=0.2),
            overall_score=0.08,
            confidence=0.8,
        ),
        "TLT": AssetScore(
            asset_ticker="TLT",
            factors=FactorScores(momentum=0.1, volatility=0.1, quality=0.4),
            overall_score=0.04,
            confidence=0.8,
        ),
        "GLD": AssetScore(
            asset_ticker="GLD",
            factors=FactorScores(momentum=0.0, volatility=0.2, quality=0.3),
            overall_score=0.03,
            confidence=0.8,
        ),
    }

    filtered, reasons = _filter_universe_by_policy(asset_scores, effective_policy)

    assert list(filtered.keys()) == ["GLD"]
    assert "QQQ" in reasons and "ограничения инвестора по сектору" in reasons["QQQ"].lower()
    assert "TLT" in reasons and "ограничения инвестора по стране" in reasons["TLT"].lower()


def test_income_preference_policy_is_enforced_in_portfolio_construction():
    effective_policy = derive_effective_policy(
        InvestorProfile(
            risk_profile=RiskProfile.moderate,
            horizon_years=10,
            target="Income",
            income_preference="income",
            constraints=Constraints(
                max_asset_weight=0.25,
                max_sector_weight=0.40,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.25,
                min_cash_weight=0.03,
                max_correlation_threshold=0.95,
            ),
        ),
        RegimeReport(
            current_regime=RegimeType.sideways,
            confidence=0.70,
            drivers=["Neutral macro"],
            is_risk_off=False,
        ),
    )
    asset_scores = {
        "AAPL": AssetScore(
            asset_ticker="AAPL",
            factors=FactorScores(momentum=0.5, volatility=-0.2, quality=0.3),
            overall_score=0.08,
            confidence=0.8,
        ),
        "MSFT": AssetScore(
            asset_ticker="MSFT",
            factors=FactorScores(momentum=0.4, volatility=-0.2, quality=0.3),
            overall_score=0.07,
            confidence=0.8,
        ),
        "TLT": AssetScore(
            asset_ticker="TLT",
            factors=FactorScores(momentum=0.1, volatility=0.2, quality=0.6),
            overall_score=0.04,
            confidence=0.8,
        ),
        "JNJ": AssetScore(
            asset_ticker="JNJ",
            factors=FactorScores(momentum=0.1, volatility=0.1, quality=0.6),
            overall_score=0.035,
            confidence=0.8,
        ),
        "GLD": AssetScore(
            asset_ticker="GLD",
            factors=FactorScores(momentum=0.0, volatility=0.2, quality=0.2),
            overall_score=0.03,
            confidence=0.8,
        ),
    }
    initial_weights = {"AAPL": 0.25, "MSFT": 0.25, "TLT": 0.22, "GLD": 0.25}

    adjusted = _apply_income_preference_policy(initial_weights, asset_scores, effective_policy)

    income_weight = adjusted.get("TLT", 0.0) + adjusted.get("JNJ", 0.0)
    growth_weight = adjusted.get("AAPL", 0.0) + adjusted.get("MSFT", 0.0)
    assert income_weight >= effective_policy.min_income_weight - 1e-6
    assert growth_weight <= effective_policy.max_growth_weight + 1e-6
