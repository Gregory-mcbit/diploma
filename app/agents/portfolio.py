import json
import math
import pandas as pd
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.agents.base import get_llm
from app.domain.asset_metadata import ASSET_CLASS_MAP, ASSET_COUNTRY_MAP, ASSET_SECTOR_MAP, ASSET_STYLE_MAP
from app.domain.schemas import AssetScore, CandidatePortfolio, Constraints, CriticVerdict, EffectivePolicy, FreshnessStatus, MemoryReference
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.policy_engine import derive_effective_policy
from app.rag.knowledge_rag import retrieve_rules
from app.tools.optimizer import (
    _cap_sector_exposure,
    _coerce_score,
    _enforce_min_cash,
    _renormalize_with_cap,
    optimize_portfolio,
    rank_allocate_portfolio,
)
from app.tools.risk_metrics import check_income_preference_alignment, check_restricted_policy_exposures
logger = get_logger(__name__)


INCOME_STYLES = {"income_defensive", "cyclical_income"}
GROWTH_STYLES = {"growth", "core_growth"}


class PortfolioRationaleOutput(BaseModel):
    sentences: List[str] = Field(min_length=3, max_length=3)


def _score_summary_for_prompt(asset_scores: Dict[str, AssetScore], selected_tickers: List[str] | None = None) -> list[dict]:
    ordered = sorted(
        asset_scores.items(),
        key=lambda item: _coerce_score(item[1]),
        reverse=True,
    )
    summary = [
        {
            "ticker": ticker,
            "overall_score": round(float(score.overall_score), 4),
            "base_model_score": round(float(score.factors.extra_factors.get("base_model_score", 0.0)), 4),
            "momentum": round(float(score.factors.momentum), 4),
            "quality": round(float(score.factors.quality), 4),
            "news_sentiment": round(float(score.factors.extra_factors.get("news_sentiment", 0.0)), 4),
            "macro_alignment": round(float(score.factors.extra_factors.get("macro_alignment", 0.0)), 4),
        }
        for ticker, score in ordered[:10]
    ]
    if not selected_tickers:
        return summary
    selected = set(selected_tickers)
    filtered_summary = [item for item in summary if item["ticker"] in selected]
    return filtered_summary or summary[: min(5, len(summary))]


def _prune_correlated_assets(
    *,
    asset_scores: Dict[str, AssetScore],
    price_df: pd.DataFrame,
    max_correlation_threshold: float,
) -> tuple[Dict[str, AssetScore], Dict[str, str]]:
    if max_correlation_threshold >= 1.0 or len(asset_scores) < 2:
        return dict(asset_scores), {}

    tickers = [ticker for ticker in asset_scores if ticker in price_df.columns]
    if len(tickers) < 2:
        return dict(asset_scores), {}

    returns = price_df[tickers].pct_change().dropna(how="all")
    if returns.empty:
        return dict(asset_scores), {}

    corr = returns.corr().fillna(0.0)
    surviving = dict(asset_scores)
    exclusion_reasons: Dict[str, str] = {}

    while True:
        active = [ticker for ticker in surviving if ticker in corr.index]
        violating_pair: tuple[str, str, float] | None = None
        for idx, left in enumerate(active):
            for right in active[idx + 1:]:
                pair_corr = float(corr.loc[left, right])
                if pair_corr > max_correlation_threshold:
                    violating_pair = (left, right, pair_corr)
                    break
            if violating_pair:
                break

        if violating_pair is None:
            break

        left, right, pair_corr = violating_pair
        left_score = _coerce_score(surviving[left])
        right_score = _coerce_score(surviving[right])
        if left_score < right_score:
            drop_ticker, keep_ticker = left, right
        elif right_score < left_score:
            drop_ticker, keep_ticker = right, left
        else:
            drop_ticker, keep_ticker = sorted([left, right])[1], sorted([left, right])[0]

        exclusion_reasons[drop_ticker] = (
            f"Исключен корреляционным фильтром, потому что {drop_ticker} слишком сильно коррелировал с "
            f"{keep_ticker} ({pair_corr:.2f} > {max_correlation_threshold:.2f}) и имел более слабый скор."
        )
        surviving.pop(drop_ticker, None)

        if not surviving:
            raise RuntimeError("Корреляционный фильтр исключил весь инвестируемый universe.")

    return surviving, exclusion_reasons

def _filter_universe_by_policy(
    asset_scores: Dict[str, AssetScore],
    effective_policy: EffectivePolicy,
) -> tuple[Dict[str, AssetScore], Dict[str, str]]:
    constraints = effective_policy.constraints
    allowed_classes = {asset_class.lower() for asset_class in constraints.allowed_asset_classes}
    forbidden_assets = {ticker.upper() for ticker in constraints.forbidden_assets}
    restricted_sectors = {sector.lower() for sector in effective_policy.restricted_sectors}
    restricted_countries = {country.lower() for country in effective_policy.restricted_countries}

    filtered: Dict[str, AssetScore] = {}
    exclusion_reasons: Dict[str, str] = {}
    for ticker, score in asset_scores.items():
        asset_class = ASSET_CLASS_MAP.get(ticker, "stocks").lower()
        sector = ASSET_SECTOR_MAP.get(ticker, "unknown").lower()
        country = ASSET_COUNTRY_MAP.get(ticker, "unknown").lower()
        if ticker.upper() in forbidden_assets:
            exclusion_reasons[ticker] = "Исключен прямым ограничением policy."
            continue
        if allowed_classes and asset_class not in allowed_classes:
            exclusion_reasons[ticker] = f"Класс актива '{asset_class}' не входит в разрешенный universe policy."
            continue
        if restricted_sectors and sector in restricted_sectors:
            exclusion_reasons[ticker] = f"Исключен из-за ограничения инвестора по сектору '{sector}'."
            continue
        if restricted_countries and country in restricted_countries:
            exclusion_reasons[ticker] = f"Исключен из-за ограничения инвестора по стране '{country}'."
            continue
        filtered[ticker] = score
    return filtered, exclusion_reasons


def _determine_allocation_mode(state: GraphState, effective_policy: EffectivePolicy) -> str:
    regime = state["market_regime"]
    explicit_mode = getattr(state["profile"].rebalancing_policy, "mode", "")
    if explicit_mode == "rank_allocation":
        return "rank_allocation"
    if effective_policy and effective_policy.income_preference == "income":
        return "income_tilt_optimizer"
    if regime.is_risk_off:
        return "regime_defensive_optimizer"
    return "mean_variance_optimizer"


def _style_weight(weights: Dict[str, float], styles: set[str]) -> float:
    return sum(weight for ticker, weight in weights.items() if ASSET_STYLE_MAP.get(ticker, "") in styles)


def _reduce_weights_pro_rata(weights: Dict[str, float], tickers: List[str], amount: float) -> Dict[str, float]:
    if amount <= 0.0:
        return dict(weights)
    donors = {ticker: weights.get(ticker, 0.0) for ticker in tickers if weights.get(ticker, 0.0) > 0.0}
    available = sum(donors.values())
    if available + 1e-8 < amount:
        raise RuntimeError("Конструкция портфеля не может высвободить достаточный вес для выполнения style-ограничений policy.")
    scale = (available - amount) / available
    adjusted = dict(weights)
    for ticker, weight in donors.items():
        adjusted[ticker] = weight * scale
    return adjusted


def _increase_weights_with_cap(
    weights: Dict[str, float],
    ordered_candidates: List[str],
    amount: float,
    max_asset_weight: float,
) -> Dict[str, float]:
    if amount <= 0.0:
        return dict(weights)
    adjusted = dict(weights)
    remaining = amount
    for ticker in ordered_candidates:
        current = adjusted.get(ticker, 0.0)
        capacity = max(0.0, max_asset_weight - current)
        if capacity <= 1e-8:
            continue
        increment = min(capacity, remaining)
        adjusted[ticker] = current + increment
        remaining -= increment
        if remaining <= 1e-8:
            break
    if remaining > 1e-6:
        raise RuntimeError("Конструкция портфеля не может разместить требуемый style-вес в рамках лимитов на актив.")
    return adjusted


def _sorted_style_candidates(asset_scores: Dict[str, AssetScore], styles: set[str]) -> List[str]:
    return sorted(
        [ticker for ticker in asset_scores if ASSET_STYLE_MAP.get(ticker, "") in styles],
        key=lambda ticker: _coerce_score(asset_scores[ticker]),
        reverse=True,
    )


def _enforce_style_floor(
    weights: Dict[str, float],
    asset_scores: Dict[str, AssetScore],
    target_styles: set[str],
    min_total_weight: float,
    max_asset_weight: float,
) -> Dict[str, float]:
    current = _style_weight(weights, target_styles)
    if current + 1e-8 >= min_total_weight:
        return dict(weights)

    candidates = _sorted_style_candidates(asset_scores, target_styles)
    required_slots = math.ceil(min_total_weight / max(max_asset_weight, 1e-8))
    if not candidates or len(candidates) < required_slots:
        raise RuntimeError("Конструкция портфеля не может выполнить минимальный style-порог policy в текущем screened universe.")

    deficit = min_total_weight - current
    non_target_tickers = [
        ticker for ticker, weight in weights.items()
        if weight > 0.0 and ASSET_STYLE_MAP.get(ticker, "") not in target_styles
    ]
    adjusted = _reduce_weights_pro_rata(weights, non_target_tickers, deficit)
    adjusted = _increase_weights_with_cap(adjusted, candidates, deficit, max_asset_weight)
    return adjusted


def _enforce_style_cap(
    weights: Dict[str, float],
    target_styles: set[str],
    max_total_weight: float,
) -> Dict[str, float]:
    current = _style_weight(weights, target_styles)
    if current <= max_total_weight + 1e-8:
        return dict(weights)
    excess = current - max_total_weight
    style_tickers = [
        ticker for ticker, weight in weights.items()
        if weight > 0.0 and ASSET_STYLE_MAP.get(ticker, "") in target_styles
    ]
    return _reduce_weights_pro_rata(weights, style_tickers, excess)


def _apply_income_preference_policy(
    weights: Dict[str, float],
    asset_scores: Dict[str, AssetScore],
    effective_policy: EffectivePolicy,
) -> Dict[str, float]:
    if effective_policy.income_preference != "income":
        return dict(weights)

    adjusted = _enforce_style_floor(
        weights=weights,
        asset_scores=asset_scores,
        target_styles=INCOME_STYLES,
        min_total_weight=effective_policy.min_income_weight,
        max_asset_weight=effective_policy.constraints.max_asset_weight,
    )
    adjusted = _enforce_style_cap(
        adjusted,
        target_styles=GROWTH_STYLES,
        max_total_weight=effective_policy.max_growth_weight,
    )
    return adjusted


def _build_construction_notes(
    *,
    allocation_mode: str,
    selected_assets: List[str],
    effective_constraints: Constraints,
    regime_name: str,
) -> List[str]:
    notes = [
        f"Режим аллокации: {allocation_mode}.",
        f"Контекст режима: {regime_name}.",
        (
            "Лимиты policy применены на уровне "
            f"{effective_constraints.max_asset_weight:.0%} на актив и "
            f"{effective_constraints.max_sector_weight:.0%} на сектор."
        ),
    ]
    if effective_constraints.min_cash_weight > 0:
        notes.append(f"Минимальный кэш-буфер зафиксирован на уровне {effective_constraints.min_cash_weight:.0%}.")
    if len(selected_assets) < 3:
        notes.append("Портфель остается концентрированным, потому что только узкий набор активов прошел policy и optimization filters.")
    return notes


def _build_uncertainty_notes(asset_scores: Dict[str, AssetScore], selected_assets: List[str]) -> List[str]:
    if not asset_scores:
        return ["Для текущего universe не удалось получить инвестируемые скоринговые оценки."]
    ordered = sorted((_coerce_score(score) for score in asset_scores.values()), reverse=True)
    notes: List[str] = []
    if ordered and max(ordered) - min(ordered) < 0.02:
        notes.append("Скоринги модели слишком близки друг к другу, поэтому относительная уверенность в выборе активов умеренная.")
    negative_count = sum(score < 0 for score in ordered)
    if negative_count >= max(1, len(ordered) // 2):
        notes.append("Заметная часть инвестируемого universe несет отрицательные сигналы по ожидаемой доходности.")
    if len(selected_assets) < min(3, len(asset_scores)):
        notes.append("Только часть screened universe получила материальный целевой вес в итоговой аллокации.")
    return notes


def _apply_revision_feedback(
    *,
    state: GraphState,
    filtered_scores: Dict[str, AssetScore],
    effective_constraints: Constraints,
    exclusion_reasons: Dict[str, str],
) -> tuple[Dict[str, AssetScore], Constraints, str | None, List[str]]:
    critic_report = state.get("critic_report")
    backtest_result = state.get("backtest_result")
    revision_count = int(state.get("revision_count", 0))
    notes: List[str] = []
    adjusted_scores = dict(filtered_scores)
    adjusted_constraints = effective_constraints.model_copy(deep=True)
    allocation_override: str | None = None

    if not critic_report or revision_count <= 0:
        return adjusted_scores, adjusted_constraints, allocation_override, notes

    if critic_report.verdict == CriticVerdict.replace_assets and backtest_result:
        weakest_assets = sorted(
            adjusted_scores,
            key=lambda ticker: _coerce_score(adjusted_scores[ticker]),
        )
        drop_count = min(max(1, len(weakest_assets) // 6), max(0, len(weakest_assets) - 4))
        for ticker in weakest_assets[:drop_count]:
            exclusion_reasons[ticker] = (
                "Исключен после критики бэктеста, потому что предыдущая конструкция отставала от рынка "
                "и этот актив входил в нижнюю часть ranking."
            )
            adjusted_scores.pop(ticker, None)
        allocation_override = "rank_allocation"
        notes.append(
            "После отставания от бенчмарка weakest assets были исключены, а аллокация переключена на более жесткий ranking mode."
        )

    if critic_report.verdict == CriticVerdict.reduce_risk and backtest_result:
        stronger_cash_floor = min(0.70, max(adjusted_constraints.min_cash_weight, 0.25 + 0.10 * revision_count))
        adjusted_constraints = adjusted_constraints.model_copy(
            update={"min_cash_weight": stronger_cash_floor}
        )
        positive_scores = {
            ticker: score for ticker, score in adjusted_scores.items() if _coerce_score(score) > 0.0
        }
        if len(positive_scores) >= 3:
            adjusted_scores = positive_scores
        allocation_override = "rank_allocation"
        notes.append(
            "После отрицательного бэктеста стратегия ужесточена: повышен cash floor и сохранены только активы с положительным скором, если их достаточно."
        )

    if not adjusted_scores:
        raise RuntimeError("После применения revision feedback у Portfolio Agent не осталось инвестируемых активов.")

    return adjusted_scores, adjusted_constraints, allocation_override, notes


def run_portfolio_agent(state: GraphState) -> Dict[str, Any]:
    """
    Portfolio Agent: The Core Builder.
    1. Queries Knowledge RAG for strict mathematical constraints.
    2. Runs Quantitative optimizer (PyPortfolioOpt) to compute final weights.
    3. LLM ONLY generates the narrative rationale (no math, no weights from LLM).
    4. CandidatePortfolio is assembled deterministically in Python code.
    """
    logger.info("Running Portfolio Agent.")

    profile = state.get("profile")
    regime = state.get("market_regime")
    effective_policy = state.get("effective_policy")
    asset_scores = state.get("asset_scores")
    parquet_pointer = state.get("market_data_pointer")

    if not profile or not regime or not asset_scores or not parquet_pointer:
        raise ValueError("Portfolio Agent требует profile, regime, asset_scores и market_data_pointer.")

    if effective_policy is None:
        effective_policy = derive_effective_policy(profile, regime)

    methodology_query = (
        f"Portfolio construction methodology for profile={profile.risk_profile.value}, "
        f"regime={regime.current_regime.value}, income_preference={effective_policy.income_preference or 'none'}, "
        f"max_asset_weight={effective_policy.constraints.max_asset_weight:.2f}, "
        f"min_cash_weight={effective_policy.constraints.min_cash_weight:.2f}, "
        f"max_correlation_threshold={effective_policy.constraints.max_correlation_threshold:.2f}"
    )
    methodology_result = retrieve_rules(methodology_query, k=5)
    methodology_context = "\n".join(
        [f"- {summary}" for summary in effective_policy.applied_rule_summaries]
        + [f"- [{match.category}] {match.content}" for match in methodology_result.matches]
    ) or "- Методологические правила не были найдены."
    effective_constraints = effective_policy.constraints
    logger.info(
        "Effective constraints: max_asset=%s max_sector=%s min_cash=%s.",
        f"{effective_constraints.max_asset_weight:.0%}",
        f"{effective_constraints.max_sector_weight:.0%}",
        f"{effective_constraints.min_cash_weight:.0%}",
    )

    # --- STEP 3: Load parquet and run deterministic PyPortfolioOpt ---
    price_df = pd.read_parquet(parquet_pointer)

    policy_filtered_scores, exclusion_reasons = _filter_universe_by_policy(asset_scores, effective_policy)
    valid_tickers = [t for t in policy_filtered_scores.keys() if t in price_df.columns]
    filtered_scores = {t: policy_filtered_scores[t] for t in valid_tickers}
    for ticker in policy_filtered_scores:
        if ticker not in price_df.columns:
            exclusion_reasons[ticker] = "Исключен, потому что для него не нашлось согласованной ценовой истории."
    if not valid_tickers:
        raise RuntimeError("У Portfolio Agent не осталось инвестируемых активов после policy-фильтрации.")
    filtered_df = price_df[valid_tickers]

    filtered_scores, correlation_exclusions = _prune_correlated_assets(
        asset_scores=filtered_scores,
        price_df=filtered_df,
        max_correlation_threshold=effective_constraints.max_correlation_threshold,
    )
    exclusion_reasons.update(correlation_exclusions)
    valid_tickers = list(filtered_scores.keys())
    if not valid_tickers:
        raise RuntimeError("У Portfolio Agent не осталось инвестируемых активов после корреляционного скрининга.")
    filtered_df = filtered_df[valid_tickers]

    filtered_scores, effective_constraints, allocation_override, revision_notes = _apply_revision_feedback(
        state=state,
        filtered_scores=filtered_scores,
        effective_constraints=effective_constraints,
        exclusion_reasons=exclusion_reasons,
    )
    valid_tickers = list(filtered_scores.keys())
    filtered_df = filtered_df[valid_tickers]

    allocation_mode = allocation_override or _determine_allocation_mode(state, effective_policy)
    if allocation_mode == "rank_allocation":
        optimized_weights = rank_allocate_portfolio(
            scores=filtered_scores,
            constraints=effective_constraints,
        )
    else:
        optimized_weights = optimize_portfolio(
            price_df=filtered_df,
            scores=filtered_scores,
            constraints=effective_constraints,
        )
        if allocation_mode == "income_tilt_optimizer":
            income_candidates = {
                ticker for ticker in optimized_weights
                if ASSET_STYLE_MAP.get(ticker) in {"income_defensive", "cyclical_income"}
            }
            if income_candidates:
                adjusted = dict(optimized_weights)
                for ticker in list(adjusted.keys()):
                    if ticker in income_candidates:
                        adjusted[ticker] *= 1.10
                    elif ASSET_STYLE_MAP.get(ticker) == "growth":
                        adjusted[ticker] *= 0.90
                total = sum(adjusted.values())
                if total <= 0:
                    raise RuntimeError("Income tilt optimizer получил неположительную сумму весов.")
                adjusted = {ticker: weight / total for ticker, weight in adjusted.items()}
                optimized_weights = adjusted
    if effective_policy.income_preference == "income":
        optimized_weights = _apply_income_preference_policy(
            weights=optimized_weights,
            asset_scores=filtered_scores,
            effective_policy=effective_policy,
        )
    optimized_weights = _renormalize_with_cap(optimized_weights, effective_constraints.max_asset_weight)
    optimized_weights = _cap_sector_exposure(optimized_weights, effective_constraints.max_sector_weight)
    optimized_weights = _enforce_min_cash(optimized_weights, effective_constraints.min_cash_weight)
    if effective_policy.income_preference == "income":
        optimized_weights = _apply_income_preference_policy(
            weights=optimized_weights,
            asset_scores=filtered_scores,
            effective_policy=effective_policy,
        )
        optimized_weights = _renormalize_with_cap(optimized_weights, effective_constraints.max_asset_weight)
        optimized_weights = _cap_sector_exposure(optimized_weights, effective_constraints.max_sector_weight)
        optimized_weights = _enforce_min_cash(optimized_weights, effective_constraints.min_cash_weight)

    restriction_violations = check_restricted_policy_exposures(
        weights=optimized_weights,
        restricted_sectors=effective_policy.restricted_sectors,
        restricted_countries=effective_policy.restricted_countries,
    )
    income_violations = check_income_preference_alignment(
        weights=optimized_weights,
        income_preference=effective_policy.income_preference,
        min_income_weight=effective_policy.min_income_weight,
        max_growth_weight=effective_policy.max_growth_weight,
    )
    policy_violations = restriction_violations + income_violations
    if policy_violations:
        raise RuntimeError(
            "Конструкция портфеля сформировала веса, нарушающие effective policy: "
            + "; ".join(policy_violations)
        )
    logger.info("Computed weights for %s assets using allocation mode %s.", len(optimized_weights), allocation_mode)

    selected = [ticker for ticker, weight in optimized_weights.items() if weight > 0.005]
    cash_weight = max(0.0, round(1.0 - sum(optimized_weights.values()), 6))
    score_summary = _score_summary_for_prompt(filtered_scores, selected)

    # --- STEP 4: LLM generates ONLY the rationale text ---
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(PortfolioRationaleOutput, method="function_calling")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты количественный портфельный управляющий и пишешь клиентское объяснение на русском языке.
Детерминированный Python-оптимизатор уже вычислил точные веса портфеля. Не предлагай и не меняй веса.

Твоя единственная задача — вернуть JSON строго по схеме с 3 короткими предложениями, объясняющими,
почему такая аллокация выглядит разумной для клиента с учетом его риск-профиля, текущего рыночного режима
и институциональных ограничений ниже.

Институциональные правила и methodology (RAG):
{methodology}
"""),
        ("human",
         "Профиль: {profile}\nРежим: {regime}\nРассчитанные веса: {weights}\n"
         "Сводка XGBoost и score-факторов: {score_summary}\n\nНапиши объяснение в 3 предложениях.")
    ])

    rationale_output: PortfolioRationaleOutput = (prompt | structured_llm).invoke({
        "methodology": methodology_context,
        "profile": profile.risk_profile.value,
        "regime": regime.current_regime.value,
        "weights": json.dumps(optimized_weights, indent=2),
        "score_summary": json.dumps(score_summary, ensure_ascii=True),
    })
    rationale_lines = rationale_output.sentences

    # --- STEP 5: Assemble CandidatePortfolio fully in Python (no LLM touching math) ---
    for ticker in valid_tickers:
        if ticker not in selected and ticker not in exclusion_reasons:
            exclusion_reasons[ticker] = "Исключен на этапе конструкции, потому что более сильные кандидаты получили приоритет в аллокации."
    construction_notes = _build_construction_notes(
        allocation_mode=allocation_mode,
        selected_assets=selected,
        effective_constraints=effective_constraints,
        regime_name=regime.current_regime.value,
    )
    construction_notes.extend(revision_notes)
    uncertainty_notes = _build_uncertainty_notes(filtered_scores, selected)

    candidate = CandidatePortfolio(
        selected_assets=selected,
        weights=optimized_weights,
        cash_weight=cash_weight,
        rationale=rationale_lines,
        allocation_mode=allocation_mode,
        construction_notes=construction_notes,
        exclusion_reasons=exclusion_reasons,
        uncertainty_notes=uncertainty_notes,
    )

    logger.info("Portfolio assembled with %s active assets and cash=%s.", len(selected), f"{cash_weight:.2%}")
    provenance = dict(state.get("provenance", {}))
    provenance["proposed_portfolio"] = build_provenance_record(
        source="pypfopt_optimizer",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.82,
        details={
            "selected_assets": len(selected),
            "cash_weight": cash_weight,
            "allocation_mode": allocation_mode,
        },
    )
    provenance["knowledge_methodology_portfolio"] = build_provenance_record(
        source=methodology_result.source,
        staleness_status=FreshnessStatus.fresh,
        confidence=0.88,
        retrieval_id=methodology_result.retrieval_id,
        details={"match_count": len(methodology_result.matches)},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["proposed_portfolio"] = FreshnessStatus.fresh.value
    freshness_map["knowledge_methodology_portfolio"] = FreshnessStatus.fresh.value
    memory_refs = list(state.get("memory_refs", []))
    memory_refs.append(
        MemoryReference(
            layer="knowledge_base",
            retrieval_id=methodology_result.retrieval_id,
            summary=f"Для portfolio stage извлечено {len(methodology_result.matches)} методологических правила.",
            source=methodology_result.source,
        )
    )
    decision_log = append_decision_log(
        state,
        stage="portfolio",
        event="portfolio_constructed",
        message=f"Сконструирован кандидатный портфель с {len(selected)} выбранными активами.",
        tool_calls=["retrieve_rules", "optimize_portfolio"],
        rule_ids=effective_policy.applied_rule_ids,
        metadata={
            "cash_weight": cash_weight,
            "selected_assets": selected,
            "allocation_mode": allocation_mode,
            "knowledge_matches": len(methodology_result.matches),
        },
    )
    return {
        "proposed_portfolio": candidate,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "memory_refs": memory_refs,
        "decision_log": decision_log,
    }
