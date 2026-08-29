from __future__ import annotations

from app.domain.schemas import Constraints, EffectivePolicy, InvestorProfile, PolicyRule, RegimeReport
from app.rag.structured_policy_rules import get_applicable_policy_rules


def _intersect_allowed_asset_classes(current: list[str], incoming: list[str]) -> list[str]:
    if not incoming:
        return list(current)
    if not current:
        return list(incoming)
    allowed = {asset_class.lower() for asset_class in current}
    return [asset_class for asset_class in incoming if asset_class.lower() in allowed]


def _normalize_string_list(values: list[str]) -> list[str]:
    normalized = {value.strip().lower() for value in values if value and value.strip()}
    return sorted(normalized)


def derive_effective_policy(
    profile: InvestorProfile,
    regime: RegimeReport,
    rules: list[PolicyRule] | None = None,
) -> EffectivePolicy:
    applicable_rules = rules or get_applicable_policy_rules(
        risk_profile=profile.risk_profile,
        regime=regime.current_regime,
    )

    constraints = Constraints(**profile.constraints.model_dump())
    restricted_sectors = _normalize_string_list(profile.sector_restrictions)
    restricted_countries = _normalize_string_list(profile.country_restrictions)
    income_preference = (profile.income_preference or "").strip().lower() or None
    min_income_weight = 0.0
    max_growth_weight = 1.0
    applied_rule_ids: list[str] = []
    applied_rule_summaries: list[str] = []

    for rule in applicable_rules:
        applied_rule_ids.append(rule.id)
        applied_rule_summaries.append(rule.description)

        if rule.max_asset_weight is not None:
            constraints.max_asset_weight = min(constraints.max_asset_weight, rule.max_asset_weight)
        if rule.max_sector_weight is not None:
            constraints.max_sector_weight = min(constraints.max_sector_weight, rule.max_sector_weight)
        if rule.min_cash_weight is not None:
            constraints.min_cash_weight = max(constraints.min_cash_weight, rule.min_cash_weight)
        if rule.max_drawdown_tolerance is not None:
            constraints.max_drawdown_tolerance = min(
                constraints.max_drawdown_tolerance,
                rule.max_drawdown_tolerance,
            )
        if rule.max_correlation_threshold is not None:
            constraints.max_correlation_threshold = min(
                constraints.max_correlation_threshold,
                rule.max_correlation_threshold,
            )
        if rule.allowed_asset_classes:
            constraints.allowed_asset_classes = _intersect_allowed_asset_classes(
                constraints.allowed_asset_classes,
                rule.allowed_asset_classes,
            )
        if rule.forbidden_assets:
            forbidden = set(constraints.forbidden_assets)
            forbidden.update(rule.forbidden_assets)
            constraints.forbidden_assets = sorted(forbidden)

    if profile.investment_amount is not None and profile.investment_amount >= 1_000_000:
        constraints.max_asset_weight = min(constraints.max_asset_weight, 0.15)
        constraints.max_sector_weight = min(constraints.max_sector_weight, 0.30)
        applied_rule_ids.append("profile-large-cap-diversification")
        applied_rule_summaries.append(
            "Крупный размер капитала ужесточил лимиты концентрации для лучшей диверсификации."
        )

    if income_preference == "income":
        constraints.min_cash_weight = max(constraints.min_cash_weight, 0.03)
        min_income_weight = max(min_income_weight, 0.35)
        max_growth_weight = min(max_growth_weight, 0.30)
        applied_rule_ids.append("profile-income-cash-buffer")
        applied_rule_summaries.append(
            "Income preference потребовал повышенный кэш-буфер и минимальную долю income-экспозиции."
        )

    if restricted_sectors:
        applied_rule_ids.append("profile-sector-restrictions")
        applied_rule_summaries.append(
            f"Портфель исключает запрещенные сектора: {', '.join(restricted_sectors)}."
        )

    if restricted_countries:
        applied_rule_ids.append("profile-country-restrictions")
        applied_rule_summaries.append(
            f"Портфель исключает запрещенные страны: {', '.join(restricted_countries)}."
        )

    return EffectivePolicy(
        constraints=constraints,
        restricted_sectors=restricted_sectors,
        restricted_countries=restricted_countries,
        income_preference=income_preference,
        min_income_weight=min_income_weight,
        max_growth_weight=max_growth_weight,
        applied_rule_ids=applied_rule_ids,
        applied_rule_summaries=applied_rule_summaries,
    )
