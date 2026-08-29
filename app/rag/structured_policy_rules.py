from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import TypeAdapter

from app.domain.schemas import PolicyRule, RegimeType, RiskProfile

POLICY_RULES_PATH = Path(__file__).with_name("policy_rules.json")
_POLICY_RULES_ADAPTER = TypeAdapter(list[PolicyRule])


def _validate_unique_rule_ids(rules: list[PolicyRule]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for rule in rules:
        if rule.id in seen:
            duplicates.append(rule.id)
        seen.add(rule.id)
    if duplicates:
        duplicate_ids = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate policy rule ids found: {duplicate_ids}")


@lru_cache(maxsize=1)
def load_policy_rules() -> list[PolicyRule]:
    raw = json.loads(POLICY_RULES_PATH.read_text(encoding="utf-8"))
    rules = _POLICY_RULES_ADAPTER.validate_python(raw)
    _validate_unique_rule_ids(rules)
    return sorted(rules, key=lambda rule: rule.priority, reverse=True)


def get_applicable_policy_rules(
    risk_profile: RiskProfile,
    regime: RegimeType,
) -> list[PolicyRule]:
    applicable = []
    for rule in load_policy_rules():
        profile_match = not rule.risk_profiles or risk_profile in rule.risk_profiles
        regime_match = not rule.regimes or regime in rule.regimes
        if profile_match and regime_match:
            applicable.append(rule)
    return applicable
