import datetime
from typing import Dict, Any
from app.domain.asset_metadata import STANDARD_UNIVERSE
from app.domain.schemas import FreshnessStatus, MemoryReference
from app.observability.logger import get_logger
from app.graph.telemetry import append_decision_log, build_provenance_record
from app.graph.state import GraphState
from app.tools.market_data import fetch_market_data
from app.tools.macro_data import fetch_macro_snapshot
from app.tools.news_retrieval import build_news_digest, fetch_latest_news
logger = get_logger(__name__)

# 400 calendar days ≈ ~280 trading days, enough for:
# - 6M momentum (mom_6m = 126 bars)
# - 60d rolling vol window
# - MACD slow=26 + signal=9 warm-up
LOOKBACK_DAYS = 400

def run_data_agent(state: GraphState) -> Dict[str, Any]:
    """
    Data Agent: Fetches the asset universe and macro data needed for scoring.

    - Downloads LOOKBACK_DAYS of price history (enough for all feature windows).
    - Returns a parquet pointer stored in data/cache/ for downstream agents.
    """
    logger.info("Running Data Agent.")

    today = datetime.datetime.now()
    start = today - datetime.timedelta(days=LOOKBACK_DAYS)

    start_str = start.strftime("%Y-%m-%d")
    end_str   = today.strftime("%Y-%m-%d")

    logger.info("Fetching %s days of data (%s to %s).", LOOKBACK_DAYS, start_str, end_str)
    logger.info("Universe: %s", STANDARD_UNIVERSE)

    parquet_pointer = fetch_market_data(
        STANDARD_UNIVERSE,
        start_date=start_str,
        end_date=end_str,
    )
    macro_data = fetch_macro_snapshot()
    news_articles, news_cache_lookups = fetch_latest_news(STANDARD_UNIVERSE, max_articles_per_ticker=2)
    news_digest = build_news_digest(news_articles)

    logger.info(
        "Data bundle cached. Universe=%s symbols. Pointer suffix=%s",
        len(STANDARD_UNIVERSE),
        parquet_pointer[-30:],
    )

    provenance = dict(state.get("provenance", {}))
    provenance["current_universe"] = build_provenance_record(
        source="standard_universe_registry",
        staleness_status=FreshnessStatus.fresh,
        confidence=1.0,
        details={"ticker_count": len(STANDARD_UNIVERSE)},
    )
    provenance["market_data_pointer"] = build_provenance_record(
        source="yfinance.market_data",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.95,
        details={
            "lookback_days": LOOKBACK_DAYS,
            "universe_size": len(STANDARD_UNIVERSE),
        },
    )
    provenance["macro_data"] = build_provenance_record(
        source=macro_data.source or "yfinance.macro_snapshot",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.90,
        details={"signal_count": len(macro_data.values)},
    )
    provenance["news_articles"] = build_provenance_record(
        source="yfinance.news",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.70,
        details={
            "ticker_count": len(news_articles),
            "article_count": news_digest.article_count,
        },
    )
    provenance["news_digest"] = build_provenance_record(
        source="news_digest_builder",
        staleness_status=FreshnessStatus.fresh,
        confidence=0.85,
        details={
            "ticker_count": len(news_digest.tickers),
            "article_count": news_digest.article_count,
        },
    )
    freshness_map = dict(state.get("freshness_map", {}))
    freshness_map["current_universe"] = FreshnessStatus.fresh.value
    freshness_map["market_data_pointer"] = FreshnessStatus.fresh.value
    freshness_map["macro_data"] = FreshnessStatus.fresh.value
    freshness_map["news_articles"] = FreshnessStatus.fresh.value
    freshness_map["news_digest"] = FreshnessStatus.fresh.value
    memory_refs = list(state.get("memory_refs", []))
    for ticker, lookup in news_cache_lookups.items():
        memory_refs.append(
            MemoryReference(
                layer="news_cache",
                retrieval_id=lookup.retrieval_id,
                summary=(
                    f"News cache hit for {ticker}."
                    if lookup.hit
                    else f"News cache {lookup.freshness_status.value} lookup for {ticker}."
                ),
                source="news_cache_rag",
            )
        )
    decision_log = append_decision_log(
        state,
        stage="data",
        event="market_data_bundle_loaded",
        message=f"Loaded market, macro, and news inputs for {len(STANDARD_UNIVERSE)} assets.",
        tool_calls=["fetch_market_data", "fetch_macro_snapshot", "fetch_latest_news"],
        metadata={
            "lookback_days": LOOKBACK_DAYS,
            "pointer_suffix": parquet_pointer[-30:],
        },
    )

    return {
        "current_universe": STANDARD_UNIVERSE,
        "market_data_pointer": parquet_pointer,
        "macro_data": macro_data,
        "news_articles": news_articles,
        "news_digest": news_digest,
        "provenance": provenance,
        "freshness_map": freshness_map,
        "memory_refs": memory_refs,
        "decision_log": decision_log,
    }
