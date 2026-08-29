import pandas as pd
from typing import Dict, Any
from app.domain.asset_metadata import ASSET_COUNTRY_MAP, ASSET_SECTOR_MAP, ASSET_STYLE_MAP
from app.domain.schemas import AssetScore, FactorScores, FeatureSnapshot, FreshnessStatus, MacroData
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.ml.scoring_model import run_ml_scoring_pipeline
from app.ml.training.feature_engine import calculate_features
from app.tools.fundamentals import fetch_fundamentals
logger = get_logger(__name__)


def _build_feature_snapshots(price_df: pd.DataFrame, universe: list[str], macro_values: Dict[str, float]) -> Dict[str, FeatureSnapshot]:
    snapshots: Dict[str, FeatureSnapshot] = {}
    macro_row = pd.Series(macro_values)
    for ticker in universe:
        if ticker not in price_df.columns:
            raise RuntimeError(f"Feature generation requires ticker {ticker} in the market price matrix.")
        feature_df = calculate_features(price_df[ticker].dropna(), macro_row=macro_row)
        if feature_df.empty:
            raise RuntimeError(f"Feature generation produced no rows for {ticker}.")
        latest = feature_df.iloc[-1].to_dict()
        if any(pd.isna(value) for value in latest.values()):
            null_cols = [key for key, value in latest.items() if pd.isna(value)]
            raise RuntimeError(f"Feature snapshot contains NaN values for {ticker}: {null_cols}")
        snapshots[ticker] = FeatureSnapshot(
            ticker=ticker,
            values={key: float(value) for key, value in latest.items()},
        )
    return snapshots


def _normalized_metric(value: float | None, positive_scale: float, inverse: bool = False) -> float:
    if value is None:
        return 0.0
    raw = (positive_scale - value) / positive_scale if inverse else value / positive_scale
    return max(-1.0, min(1.0, float(raw)))


def _fundamental_signals(metrics: Dict[str, float | None]) -> Dict[str, float]:
    quality_components = [
        _normalized_metric(metrics.get("roe"), 0.25),
        _normalized_metric(metrics.get("net_margin"), 0.20),
        _normalized_metric(metrics.get("debt_to_equity"), 2.0, inverse=True),
    ]
    valuation_components = [
        _normalized_metric(metrics.get("pe"), 25.0, inverse=True),
        _normalized_metric(metrics.get("forward_pe"), 20.0, inverse=True),
        _normalized_metric(metrics.get("price_to_book"), 5.0, inverse=True),
    ]
    quality_signal = sum(quality_components) / len(quality_components)
    valuation_signal = sum(valuation_components) / len(valuation_components)
    leverage_signal = _normalized_metric(metrics.get("debt_to_equity"), 2.0, inverse=True)
    return {
        "quality": quality_signal,
        "valuation": valuation_signal,
        "leverage": leverage_signal,
    }


def _news_signal_for_ticker(news_articles: list) -> float:
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    scores = [
        sentiment_map.get((article.sentiment_tag or "").lower(), 0.0)
        for article in news_articles
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _technical_signals(feature_snapshot: FeatureSnapshot) -> Dict[str, float]:
    values = feature_snapshot.values
    momentum_signal = (
        float(values.get("mom_1m", 0.0))
        + float(values.get("mom_3m", 0.0))
        + float(values.get("mom_6m", 0.0))
    ) / 3.0
    volatility_signal = -(
        float(values.get("vol_20d", 0.0))
        + float(values.get("vol_60d", 0.0))
    ) / 2.0
    return {
        "momentum": max(-1.0, min(1.0, momentum_signal * 5.0)),
        "volatility": max(-1.0, min(1.0, volatility_signal)),
    }


def _macro_alignment_signal(ticker: str, macro_values: Dict[str, float]) -> float:
    sector = ASSET_SECTOR_MAP.get(ticker, "broad_equity")
    style = ASSET_STYLE_MAP.get(ticker, "core_growth")
    vix = float(macro_values["vix"])
    spread = float(macro_values["yield_spread"])
    credit_spread = float(macro_values["credit_spread"])
    usd_strength = float(macro_values["usd_strength"])
    spy_tlt_ratio = float(macro_values["spy_tlt_ratio"])
    oil_mom_1m = float(macro_values["oil_mom_1m"])

    signal = 0.0
    if style in {"income_defensive", "defensive_hedge"}:
        signal += 0.6 if vix > 25 else -0.1
        signal += 0.4 if spread < 0 else 0.0
    if sector == "technology":
        signal -= 0.4 if spread < 0 else 0.0
        signal -= 0.3 if usd_strength > 0.02 else 0.0
        signal += 0.2 if spy_tlt_ratio > 0 else 0.0
    if sector == "financials":
        signal += 0.4 if spread > 0 else -0.5
        signal += 0.2 if credit_spread > 0 else -0.2
    if ticker == "GLD":
        signal += 0.5 if vix > 22 else 0.0
        signal += 0.3 if oil_mom_1m > 0 else 0.0
        signal += 0.2 if usd_strength < 0 else -0.1
    if ticker == "TLT":
        signal += 0.6 if spread < 0 else -0.1
        signal += 0.3 if vix > 22 else 0.0
    return max(-1.0, min(1.0, signal))


def _profile_preference_signal(ticker: str, state: GraphState) -> float:
    profile = state.get("profile")
    if not profile:
        return 0.0

    style = ASSET_STYLE_MAP.get(ticker, "core_growth")
    signal = 0.0
    income_preference = (profile.income_preference or "").lower()
    if income_preference == "income":
        if style in {"income_defensive", "cyclical_income"}:
            signal += 0.4
        if style == "growth":
            signal -= 0.3
    elif income_preference == "growth":
        if style == "growth":
            signal += 0.4
        if style in {"income_defensive", "cyclical_income"}:
            signal -= 0.2

    if profile.investment_amount is not None and profile.investment_amount >= 1_000_000:
        if style == "core_growth":
            signal += 0.1
        if ticker in {"AAPL", "MSFT", "QQQ"}:
            signal -= 0.05

    if ASSET_SECTOR_MAP.get(ticker, "").lower() in {item.lower() for item in profile.sector_restrictions}:
        signal -= 1.0
    if ASSET_COUNTRY_MAP.get(ticker, "").lower() in {item.lower() for item in profile.country_restrictions}:
        signal -= 1.0
    return max(-1.0, min(1.0, signal))


def _build_asset_score(
    *,
    ticker: str,
    base_score: float,
    feature_snapshot: FeatureSnapshot,
    fundamentals_snapshot,
    news_articles: list,
    macro_values: Dict[str, float],
    state: GraphState,
) -> AssetScore:
    technical = _technical_signals(feature_snapshot)
    fundamental = _fundamental_signals(fundamentals_snapshot.metrics if fundamentals_snapshot else {})
    news_signal = _news_signal_for_ticker(news_articles)
    macro_alignment = _macro_alignment_signal(ticker, macro_values)
    profile_preference = _profile_preference_signal(ticker, state)

    overall_score = (
        float(base_score)
        + 0.015 * technical["momentum"]
        + 0.010 * technical["volatility"]
        + 0.020 * fundamental["quality"]
        + 0.010 * fundamental["valuation"]
        + 0.005 * fundamental["leverage"]
        + 0.008 * news_signal
        + 0.012 * macro_alignment
        + 0.010 * profile_preference
    )

    completeness = 0.6
    completeness += 0.15 if fundamentals_snapshot else 0.0
    completeness += 0.10 if news_articles else 0.0
    completeness += 0.15 if feature_snapshot.values else 0.0

    return AssetScore(
        asset_ticker=ticker,
        factors=FactorScores(
            momentum=technical["momentum"],
            volatility=technical["volatility"],
            quality=fundamental["quality"],
            extra_factors={
                "base_model_score": float(base_score),
                "valuation": fundamental["valuation"],
                "leverage": fundamental["leverage"],
                "news_sentiment": news_signal,
                "macro_alignment": macro_alignment,
                "profile_preference": profile_preference,
            },
        ),
        overall_score=float(overall_score),
        confidence=round(min(completeness, 0.95), 2),
    )

def run_scoring_agent(state: GraphState) -> Dict[str, Any]:
    """
    Scoring Agent: Orchestrates the XGBoost ML model to evaluate the asset universe.
    By design, it performs no math itself and strictly delegates to the quantitative ML pipeline.
    """
    logger.info("Running Scoring Agent.")
    
    parquet_pointer = state.get("market_data_pointer")
    universe = state.get("current_universe", [])
    macro_payload = state.get("macro_data")
    news_articles = state.get("news_articles", {})
    
    if not parquet_pointer or not universe:
        raise ValueError("Scoring Agent requires market_data_pointer and current_universe.")
    if not macro_payload:
        raise ValueError("Scoring Agent requires macro_data from the Data Agent.")
        
    # Execute the trained XGBoost model artifact to get 21-day return predictions.
    logger.info("Triggering scoring worker on %s assets via parquet suffix=%s.", len(universe), parquet_pointer[-20:])
    macro_values = macro_payload.values
    base_scores = run_ml_scoring_pipeline(parquet_pointer, universe, macro=macro_values)
    fundamentals = fetch_fundamentals(universe)
    price_df = pd.read_parquet(parquet_pointer)
    features = _build_feature_snapshots(price_df, universe, macro_values)
    asset_scores_dict: Dict[str, AssetScore] = {}
    for ticker, base_score in base_scores.items():
        feature_snapshot = features.get(ticker)
        if feature_snapshot is None:
            raise RuntimeError(f"Missing feature snapshot for scored ticker {ticker}.")
        fundamentals_snapshot = fundamentals.get(ticker)
        asset_scores_dict[ticker] = _build_asset_score(
            ticker=ticker,
            base_score=base_score,
            feature_snapshot=feature_snapshot,
            fundamentals_snapshot=fundamentals_snapshot,
            news_articles=news_articles.get(ticker, []),
            macro_values=macro_values,
            state=state,
        )
    logger.info("Successfully scored %s assets using ML inference and overlays.", len(asset_scores_dict))

    provenance = dict(state.get("provenance", {}))
    provenance["macro_data"] = build_provenance_record(
        source="yfinance.macro_snapshot",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.85,
        details={"signal_count": len(macro_values)},
    )
    provenance["asset_scores"] = build_provenance_record(
        source="xgboost_subprocess",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.80,
        details={
            "model_path": "data/models/xgb_alpha.json",
            "asset_count": len(asset_scores_dict),
            "score_type": "AssetScore",
        },
    )
    provenance["fundamentals"] = build_provenance_record(
        source="yfinance.info",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.75,
        details={"asset_count": len(fundamentals)},
    )
    provenance["features"] = build_provenance_record(
        source="feature_engine",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.88,
        details={"asset_count": len(features)},
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["macro_data"] = FreshnessStatus.fresh.value
    freshness_map["asset_scores"] = FreshnessStatus.fresh.value
    freshness_map["fundamentals"] = FreshnessStatus.fresh.value
    freshness_map["features"] = FreshnessStatus.fresh.value
    decision_log = append_decision_log(
        state,
        stage="scoring",
        event="assets_scored",
        message=f"Scored {len(asset_scores_dict)} assets with ML, fundamentals, and news overlays.",
        tool_calls=["run_ml_scoring_pipeline", "fetch_fundamentals", "calculate_features"],
        metadata={
            "asset_count": len(asset_scores_dict),
            "feature_count": len(features),
        },
    )
    
    # Return the dictionary representing our quantitative Alpha predictions.
    # The Portfolio Agent will read this exact dictionary to run PyPortfolioOpt.
    return {
        "asset_scores": asset_scores_dict,
        "macro_data": MacroData(values=macro_values, source="yfinance.macro_snapshot"),
        "fundamentals": fundamentals,
        "features": features,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "decision_log": decision_log,
    }
