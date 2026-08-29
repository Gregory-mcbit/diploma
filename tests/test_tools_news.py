import json

from app.domain.schemas import NewsArticle
from app.tools.news_retrieval import build_news_digest, fetch_latest_news


class DummyTicker:
    def __init__(self, ticker: str):
        self.news = [
            {
                "title": f"{ticker} gains on strong guidance",
                "summary": f"{ticker} posted stronger-than-expected demand.",
                "publisher": "Reuters",
                "providerPublishTime": 1712000000,
                "link": f"https://example.com/{ticker.lower()}-1",
                "sentimentTag": "positive",
            },
            {
                "title": f"{ticker} faces margin pressure",
                "summary": f"{ticker} signaled margin pressure for the next quarter.",
                "publisher": "Bloomberg",
                "providerPublishTime": 1712003600,
                "link": f"https://example.com/{ticker.lower()}-2",
                "sentimentTag": "negative",
            },
        ]


def test_fetch_latest_news_and_cache(monkeypatch):
    cache: dict[str, str] = {}

    monkeypatch.setattr(
        "app.tools.news_retrieval.check_news_cache",
        lambda ticker: type(
            "CacheLookup",
            (),
            {
                "retrieval_id": f"lookup-{ticker}",
                "ticker": ticker,
                "hit": ticker in cache,
                "freshness_status": type("Freshness", (), {"value": "fresh" if ticker in cache else "unknown"})(),
                "payload": cache.get(ticker),
            },
        )(),
    )
    monkeypatch.setattr(
        "app.tools.news_retrieval.write_news_cache",
        lambda ticker, payload, source="web": cache.__setitem__(ticker, payload),
    )
    monkeypatch.setattr("app.tools.news_retrieval.yf.Ticker", DummyTicker)

    articles, cache_lookups = fetch_latest_news(["SPY"], max_articles_per_ticker=2)

    assert len(articles["SPY"]) == 2
    assert isinstance(articles["SPY"][0], NewsArticle)
    assert json.loads(cache["SPY"])[0]["title"] == "SPY gains on strong guidance"
    assert cache_lookups["SPY"].retrieval_id == "lookup-SPY"


def test_build_news_digest():
    articles = {
        "SPY": [
            NewsArticle(
                ticker="SPY",
                source="Reuters",
                timestamp="2024-04-01T00:00:00+00:00",
                title="SPY gains on strong guidance",
                summary="SPY posted stronger-than-expected demand.",
                url="https://example.com/spy-1",
                sentiment_tag="positive",
            )
        ]
    }

    digest = build_news_digest(articles)
    assert digest.article_count == 1
    assert digest.by_ticker["SPY"].titles == ["SPY gains on strong guidance"]
    assert digest.by_ticker["SPY"].sources == ["Reuters"]
    assert digest.sentiment_totals["positive"] == 1


def test_fetch_latest_news_skips_malformed_items(monkeypatch):
    cache: dict[str, str] = {}

    monkeypatch.setattr(
        "app.tools.news_retrieval.check_news_cache",
        lambda ticker: type(
            "CacheLookup",
            (),
            {
                "retrieval_id": f"lookup-{ticker}",
                "ticker": ticker,
                "hit": False,
                "freshness_status": type("Freshness", (), {"value": "unknown"})(),
                "payload": None,
            },
        )(),
    )
    monkeypatch.setattr(
        "app.tools.news_retrieval.write_news_cache",
        lambda ticker, payload, source="web": cache.__setitem__(ticker, payload),
    )
    monkeypatch.setattr(
        "app.tools.news_retrieval.yf.Ticker",
        lambda ticker: type(
            "BrokenTicker",
            (),
            {
                "news": [
                    {
                        "summary": "Missing title should be skipped.",
                        "publisher": "Reuters",
                        "providerPublishTime": 1712000000,
                        "link": "https://example.com/bad-item",
                    },
                    {
                        "title": f"{ticker} gains on strong guidance",
                        "summary": f"{ticker} posted stronger-than-expected demand.",
                        "publisher": "Reuters",
                        "providerPublishTime": 1712003600,
                        "link": f"https://example.com/{ticker.lower()}-1",
                        "sentimentTag": "positive",
                    },
                ]
            },
        )(),
    )

    articles, _ = fetch_latest_news(["SPY"], max_articles_per_ticker=2)

    assert len(articles["SPY"]) == 1
    assert articles["SPY"][0].title == "SPY gains on strong guidance"
    assert len(json.loads(cache["SPY"])) == 1
