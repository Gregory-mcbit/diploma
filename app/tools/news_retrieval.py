from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import yfinance as yf

from app.domain.schemas import NewsArticle, NewsDigest, NewsDigestItem
from app.observability.logger import get_logger
from app.rag.news_cache_rag import check_news_cache, write_news_cache
from app.rag.schemas import NewsCacheLookup


logger = get_logger(__name__)

POSITIVE_KEYWORDS = {
    "beat", "beats", "growth", "strong", "surge", "surges", "gain", "gains",
    "bullish", "upside", "upgrade", "record", "expands", "expansion",
    "improves", "improvement", "profit", "profits", "momentum", "outperform",
}
NEGATIVE_KEYWORDS = {
    "miss", "misses", "weak", "drop", "drops", "decline", "declines", "pressure",
    "cut", "cuts", "downgrade", "warning", "warns", "loss", "losses", "slump",
    "lawsuit", "probe", "investigation", "risk", "risks", "bearish", "fall",
}


def _normalize_timestamp(value) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


def _extract_url(item: dict) -> str:
    canonical = item.get("canonicalUrl")
    if isinstance(canonical, dict) and canonical.get("url"):
        return str(canonical["url"])
    if item.get("link"):
        return str(item["link"])
    if item.get("url"):
        return str(item["url"])
    raise RuntimeError("News item is missing a URL.")


def _extract_summary(item: dict) -> str:
    if item.get("summary"):
        return str(item["summary"])
    content = item.get("content")
    if isinstance(content, dict) and content.get("summary"):
        return str(content["summary"])
    if item.get("title"):
        return str(item["title"])
    raise RuntimeError("News item is missing summary content.")


def _normalize_news_item(ticker: str, item: dict) -> NewsArticle:
    title = item.get("title")
    if not title:
        raise RuntimeError("News item is missing title.")
    publisher = item.get("publisher") or item.get("providerPublishTime") or "yfinance"
    sentiment = item.get("sentiment") or item.get("sentimentTag")
    return NewsArticle(
        ticker=ticker,
        source=str(publisher),
        timestamp=_normalize_timestamp(item.get("providerPublishTime") or item.get("pubDate")),
        title=str(title),
        summary=_extract_summary(item),
        url=_extract_url(item),
        sentiment_tag=str(sentiment) if sentiment is not None else None,
    )


def _infer_sentiment_tag(article: NewsArticle) -> str:
    explicit = (article.sentiment_tag or "").strip().lower()
    if explicit in {"positive", "negative", "neutral"}:
        return explicit

    text = f"{article.title} {article.summary}".lower()
    positive_hits = sum(1 for token in POSITIVE_KEYWORDS if token in text)
    negative_hits = sum(1 for token in NEGATIVE_KEYWORDS if token in text)
    if positive_hits > negative_hits:
        return "positive"
    if negative_hits > positive_hits:
        return "negative"
    return "neutral"


def _serialize_articles(articles: List[NewsArticle]) -> str:
    return json.dumps([article.model_dump() for article in articles], ensure_ascii=True)


def _deserialize_articles(payload: str) -> List[NewsArticle]:
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise RuntimeError("Cached news payload must be a list.")
    return [NewsArticle.model_validate(item) for item in raw]


def fetch_latest_news(
    tickers: List[str],
    max_articles_per_ticker: int = 3,
) -> Tuple[Dict[str, List[NewsArticle]], Dict[str, NewsCacheLookup]]:
    articles_by_ticker: Dict[str, List[NewsArticle]] = {}
    cache_lookups: Dict[str, NewsCacheLookup] = {}
    for ticker in tickers:
        cache_lookup = check_news_cache(ticker)
        cache_lookups[ticker] = cache_lookup
        if cache_lookup.hit:
            if not cache_lookup.payload:
                raise RuntimeError(f"Fresh news cache hit for {ticker} is missing payload.")
            articles_by_ticker[ticker] = _deserialize_articles(cache_lookup.payload)
            continue

        raw_items = yf.Ticker(ticker).news
        if raw_items is None:
            raise RuntimeError(f"Live news retrieval returned None for {ticker}.")
        if not isinstance(raw_items, list):
            raise RuntimeError(f"Live news retrieval for {ticker} must return a list of items.")

        normalized: List[NewsArticle] = []
        for item in raw_items[:max_articles_per_ticker]:
            try:
                article = _normalize_news_item(ticker, item)
                normalized.append(article.model_copy(update={"sentiment_tag": _infer_sentiment_tag(article)}))
            except RuntimeError as exc:
                logger.warning("Skipping malformed news item for %s: %s", ticker, exc)

        articles_by_ticker[ticker] = normalized
        write_news_cache(ticker, _serialize_articles(normalized), source="yfinance.news")

    logger.info("Loaded news for %s tickers.", len(articles_by_ticker))
    return articles_by_ticker, cache_lookups


def build_news_digest(
    articles_by_ticker: Dict[str, List[NewsArticle]],
    max_titles_per_ticker: int = 3,
) -> NewsDigest:
    by_ticker: Dict[str, NewsDigestItem] = {}
    sentiment_totals = {"positive": 0, "neutral": 0, "negative": 0}
    total_articles = 0

    for ticker, articles in articles_by_ticker.items():
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        titles: List[str] = []
        sources: List[str] = []
        for article in articles:
            total_articles += 1
            sentiment = (article.sentiment_tag or "neutral").lower()
            if sentiment == "positive":
                positive_count += 1
            elif sentiment == "negative":
                negative_count += 1
            else:
                neutral_count += 1
                sentiment = "neutral"
            sentiment_totals[sentiment] = sentiment_totals.get(sentiment, 0) + 1
            if len(titles) < max_titles_per_ticker:
                titles.append(article.title)
            if article.source not in sources:
                sources.append(article.source)
        by_ticker[ticker] = NewsDigestItem(
            ticker=ticker,
            article_count=len(articles),
            positive_count=positive_count,
            neutral_count=neutral_count,
            negative_count=negative_count,
            titles=titles,
            sources=sources,
        )

    return NewsDigest(
        article_count=total_articles,
        tickers=sorted(articles_by_ticker.keys()),
        sentiment_totals=sentiment_totals,
        by_ticker=by_ticker,
    )
