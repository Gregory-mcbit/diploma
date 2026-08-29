from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel

from app.api.schemas import MonitoringRunRequest, PortfolioRunResponse
from app.domain.asset_metadata import ASSET_CLASS_MAP, ASSET_COUNTRY_MAP, ASSET_REGISTRY, ASSET_SECTOR_MAP, ASSET_STYLE_MAP
from app.domain.schemas import (
    AssetScore,
    BacktestResult,
    CandidatePortfolio,
    Constraints,
    FeatureSnapshot,
    FundamentalSnapshot,
    InvestorProfile,
    NewsDigest,
    RebalancingPolicy,
    RiskProfile,
)
from app.observability.tracing import to_serializable


ASSET_NAME_MAP: Dict[str, str] = {
    "SPY": "SPDR S&P 500 ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "QQQ": "Invesco QQQ Trust",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    "IWM": "iShares Russell 2000 ETF",
    "RSP": "Invesco S&P 500 Equal Weight ETF",
    "VT": "Vanguard Total World Stock ETF",
    "ACWI": "iShares MSCI ACWI ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "VGK": "Vanguard FTSE Europe ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "FXI": "iShares China Large-Cap ETF",
    "INDA": "iShares MSCI India ETF",
    "EWC": "iShares MSCI Canada ETF",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "SMH": "VanEck Semiconductor ETF",
    "SOXX": "iShares Semiconductor ETF",
    "XBI": "SPDR S&P Biotech ETF",
    "VNQ": "Vanguard Real Estate ETF",
    "IYR": "iShares U.S. Real Estate ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "TIP": "iShares TIPS Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
    "MUB": "iShares National Muni Bond ETF",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "USO": "United States Oil Fund",
    "UNG": "United States Natural Gas Fund",
    "DBA": "Invesco DB Agriculture Fund",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "WMT": "Walmart",
    "COST": "Costco",
    "HD": "Home Depot",
    "MCD": "McDonald's",
    "NKE": "Nike",
    "CAT": "Caterpillar",
    "GE": "GE Aerospace",
    "RTX": "RTX",
    "LIN": "Linde",
    "PLD": "Prologis",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "CSCO": "Cisco",
    "QCOM": "Qualcomm",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "AMAT": "Applied Materials",
    "NFLX": "Netflix",
    "DIS": "Disney",
    "V": "Visa",
    "MA": "Mastercard",
    "TMUS": "T-Mobile US",
    "TMO": "Thermo Fisher Scientific",
    "ACN": "Accenture",
    "INTU": "Intuit",
    "IBM": "IBM",
    "BKNG": "Booking Holdings",
    "BRK-B": "Berkshire Hathaway",
    "HON": "Honeywell",
}

SECTOR_RU_MAP = {
    "broad_equity": "широкий рынок акций США",
    "technology": "технологический сектор",
    "industrials": "промышленный сектор",
    "small_cap": "малые компании США",
    "global_equity": "глобальный рынок акций",
    "developed_equity": "развитые рынки вне США",
    "emerging_markets": "развивающиеся рынки",
    "europe": "европейский рынок акций",
    "japan": "японский рынок акций",
    "china": "китайский рынок акций",
    "india": "индийский рынок акций",
    "canada": "канадский рынок акций",
    "financials": "финансовый сектор",
    "healthcare": "сектор здравоохранения",
    "energy": "энергетический сектор",
    "consumer_staples": "потребительский защитный сектор",
    "consumer_discretionary": "потребительский циклический сектор",
    "utilities": "коммунальный сектор",
    "materials": "сектор материалов",
    "real_estate": "недвижимость",
    "semiconductors": "полупроводники",
    "biotech": "биотехнологии",
    "fixed_income": "госдолг и широкий облигационный рынок США",
    "inflation_linked": "инфляционно-защищенные облигации США",
    "credit": "корпоративные облигации инвестиционного уровня",
    "high_yield": "высокодоходные корпоративные облигации",
    "emerging_debt": "долг развивающихся стран",
    "municipal_bonds": "муниципальные облигации США",
    "commodities": "драгоценные металлы и сырьевой хедж",
    "energy_commodities": "энергетические товары",
    "agriculture": "сельскохозяйственные товары",
    "communication_services": "коммуникационные сервисы",
}

STYLE_RU_MAP = {
    "core_growth": "ядро роста",
    "growth": "рост",
    "income_defensive": "защитный доходный актив",
    "cyclical_income": "циклический доходный актив",
    "defensive_hedge": "защитный хедж",
}

COUNTRY_RU_MAP = {
    "us": "США",
    "global": "глобальный рынок",
    "developed_ex_us": "развитые рынки вне США",
    "emerging_markets": "развивающиеся рынки",
    "europe": "Европа",
    "japan": "Япония",
    "china": "Китай",
    "india": "Индия",
    "canada": "Канада",
}


def _arrow_safe_cell(value: Any) -> Any:
    serializable = to_serializable(value)
    if serializable is None or isinstance(serializable, (str, int, float, bool)):
        return serializable
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def _arrow_safe_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    safe = frame.copy()
    for column in safe.columns:
        if str(safe[column].dtype) != "object":
            continue
        safe[column] = safe[column].map(_arrow_safe_cell)
        non_null = safe[column].dropna()
        if len({type(value) for value in non_null}) > 1:
            safe[column] = safe[column].map(lambda value: None if value is None else str(value))
    return safe


def split_csv_input(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_json_mapping(raw: str, field_name: str) -> Dict[str, float]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    normalized: Dict[str, float] = {}
    for key, value in parsed.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-numeric value for {key}.") from exc
    return normalized


def weights_to_dataframe(weights: Dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"Ticker": ticker, "Weight": weight} for ticker, weight in weights.items()]
    )
    if frame.empty:
        return frame
    return frame.sort_values("Weight", ascending=False).reset_index(drop=True)


def mapping_to_dataframe(mapping: Dict[str, Any], key_name: str = "Key", value_name: str = "Value") -> pd.DataFrame:
    frame = pd.DataFrame([{key_name: key, value_name: value} for key, value in mapping.items()])
    if frame.empty:
        return frame
    return _arrow_safe_dataframe(frame.reset_index(drop=True))


def list_to_dataframe(values: List[Any], column_name: str = "Value") -> pd.DataFrame:
    return _arrow_safe_dataframe(pd.DataFrame([{column_name: value} for value in values]))


def backtest_curve_dataframe(backtest: BacktestResult) -> pd.DataFrame:
    required_lengths = {
        len(backtest.curve_dates),
        len(backtest.portfolio_curve),
        len(backtest.benchmark_curve),
        len(backtest.equal_weight_curve),
    }
    if len(required_lengths) != 1 or not backtest.curve_dates:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "Дата": pd.to_datetime(backtest.curve_dates),
            "Портфель": [(value - 1.0) * 100.0 for value in backtest.portfolio_curve],
            "SPY benchmark": [(value - 1.0) * 100.0 for value in backtest.benchmark_curve],
            "Equal weight": [(value - 1.0) * 100.0 for value in backtest.equal_weight_curve],
        }
    )
    return frame.set_index("Дата")


def backtest_drawdown_dataframe(backtest: BacktestResult) -> pd.DataFrame:
    if not backtest.curve_dates or len(backtest.curve_dates) != len(backtest.drawdown_curve):
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "Дата": pd.to_datetime(backtest.curve_dates),
            "Просадка портфеля": [value * 100.0 for value in backtest.drawdown_curve],
        }
    )
    return frame.set_index("Дата")


def model_to_pretty_json(value: Any) -> str:
    serializable = to_serializable(value)
    return json.dumps(serializable, indent=2, ensure_ascii=False)


def trace_log_to_dataframe(trace_log: List[Any]) -> pd.DataFrame:
    rows = [to_serializable(item) for item in trace_log]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    preferred_columns = [
        "timestamp",
        "request_id",
        "correlation_id",
        "stage",
        "event",
        "message",
        "metadata",
    ]
    return _arrow_safe_dataframe(frame[[col for col in preferred_columns if col in frame.columns]])


def decision_log_to_dataframe(decision_log: List[Any]) -> pd.DataFrame:
    rows = [to_serializable(item) for item in decision_log]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    preferred_columns = [
        "timestamp",
        "stage",
        "event",
        "message",
        "tool_calls",
        "rule_ids",
        "metadata",
    ]
    return _arrow_safe_dataframe(frame[[col for col in preferred_columns if col in frame.columns]])


def format_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "н/д"
    return f"{float(value):.{digits}%}"


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "н/д"
    return f"{float(value):.{digits}f}"


def asset_display_name(ticker: str) -> str:
    return ASSET_NAME_MAP.get(ticker, ticker)


def asset_description(ticker: str) -> str:
    metadata = ASSET_REGISTRY.get(ticker)
    if not metadata:
        return f"{ticker} — финансовый инструмент из текущего инвестиционного universe."

    asset_class = metadata["asset_class"]
    sector = SECTOR_RU_MAP.get(metadata["sector"], metadata["sector"])
    style = STYLE_RU_MAP.get(metadata["style"], metadata["style"])
    country = COUNTRY_RU_MAP.get(metadata["country"], metadata["country"])
    name = asset_display_name(ticker)

    if ticker in ASSET_NAME_MAP and ticker not in {
        "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "RSP", "VT", "ACWI", "EFA", "IEFA", "EEM", "VEA",
        "VWO", "VGK", "EWJ", "FXI", "INDA", "EWC", "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU",
        "XLB", "XLRE", "SMH", "SOXX", "XBI", "VNQ", "IYR", "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "BND",
        "AGG", "EMB", "MUB", "GLD", "SLV", "USO", "UNG", "DBA",
    }:
        return f"{name} ({ticker}) — крупная публичная компания, представляющая {sector.lower()} в регионе {country.lower()}. По стилю это {style.lower()}."

    if asset_class == "stocks":
        return f"{name} ({ticker}) — биржевой инструмент на {sector.lower()} с географией {country.lower()}. В текущей модели это {style.lower()}."
    if asset_class == "bonds":
        return f"{name} ({ticker}) — облигационный инструмент, дающий доступ к сегменту «{sector.lower()}». Обычно используется как более стабильная часть портфеля."
    if asset_class == "commodities":
        return f"{name} ({ticker}) — сырьевой инструмент на тему «{sector.lower()}», который часто добавляют как защитный или диверсифицирующий слой."
    return f"{name} ({ticker}) — инструмент класса {asset_class}, связанный с темой «{sector.lower()}»."


def asset_selection_summary(response: PortfolioRunResponse, ticker: str) -> str:
    description = asset_description(ticker)
    inclusion_reason = response.final_report.inclusion_reasons.get(
        ticker,
        "Актив прошел фильтры политики и получил достаточно сильный суммарный скор для попадания в портфель.",
    )
    weight = format_pct(response.final_report.portfolio.weights.get(ticker), 1)
    return f"{description} Текущий вес в портфеле: {weight}. Почему выбран: {inclusion_reason}"


def selected_assets_summary_dataframe(response: PortfolioRunResponse) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in response.final_report.portfolio.selected_assets:
        rows.append(
            {
                "Тикер": ticker,
                "Название": asset_display_name(ticker),
                "Вес": format_pct(response.final_report.portfolio.weights.get(ticker), 1),
                "Что это": asset_description(ticker),
                "Почему выбрано": response.final_report.inclusion_reasons.get(
                    ticker,
                    "Актив прошел фильтры политики и получил достаточно сильный суммарный скор для попадания в портфель.",
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return _arrow_safe_dataframe(frame)


def asset_overview_dataframe(response: PortfolioRunResponse) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    news_digest: NewsDigest | None = response.news_digest
    for ticker in response.final_report.portfolio.selected_assets:
        score: AssetScore | None = response.asset_scores.get(ticker)
        fundamentals: FundamentalSnapshot | None = response.fundamentals.get(ticker)
        news_item = news_digest.by_ticker.get(ticker) if news_digest else None
        rows.append(
            {
                "Тикер": ticker,
                "Название": asset_display_name(ticker),
                "Вес": format_pct(response.final_report.portfolio.weights.get(ticker), 1),
                "Скор": format_float(score.overall_score if score else None, 3),
                "Уверенность": format_pct(score.confidence if score else None, 0),
                "Класс": ASSET_CLASS_MAP.get(ticker, "н/д"),
                "Сектор": ASSET_SECTOR_MAP.get(ticker, "н/д"),
                "Страна": ASSET_COUNTRY_MAP.get(ticker, "н/д"),
                "Стиль": ASSET_STYLE_MAP.get(ticker, "н/д"),
                "Что это": asset_description(ticker),
                "Почему выбрано": response.final_report.inclusion_reasons.get(
                    ticker,
                    "Актив прошел фильтры политики и получил достаточно сильный суммарный скор для попадания в портфель.",
                ),
                "P/E": format_float((fundamentals.metrics.get("pe") if fundamentals else None), 1),
                "ROE": format_pct((fundamentals.metrics.get("roe") if fundamentals else None), 1),
                "Новостей": news_item.article_count if news_item else 0,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return _arrow_safe_dataframe(frame)


def asset_factor_dataframe(score: AssetScore | None) -> pd.DataFrame:
    if score is None:
        return pd.DataFrame()
    factors = {
        "momentum": score.factors.momentum,
        "volatility": score.factors.volatility,
        "quality": score.factors.quality,
        **score.factors.extra_factors,
    }
    rows = [{"Фактор": key, "Значение": format_float(value, 3)} for key, value in factors.items()]
    return pd.DataFrame(rows)


def fundamentals_dataframe(snapshot: FundamentalSnapshot | None) -> pd.DataFrame:
    if snapshot is None:
        return pd.DataFrame()
    rows = [
        {"Метрика": key, "Значение": format_float(value, 3) if value is not None else "н/д"}
        for key, value in snapshot.metrics.items()
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return _arrow_safe_dataframe(frame)


def feature_snapshot_dataframe(snapshot: FeatureSnapshot | None) -> pd.DataFrame:
    if snapshot is None:
        return pd.DataFrame()
    preferred_order = [
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "vol_20d",
        "vol_60d",
        "rsi_14",
        "macd_hist",
        "bb_pct",
        "vix",
        "yield_10y",
        "yield_3m",
        "yield_spread",
        "credit_spread",
        "usd_strength",
        "oil_mom_1m",
        "spy_tlt_ratio",
    ]
    rows = [{"Признак": key, "Значение": format_float(snapshot.values.get(key), 4)} for key in preferred_order if key in snapshot.values]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return _arrow_safe_dataframe(frame)


def build_monitoring_request_from_portfolio_response(
    response: PortfolioRunResponse,
    profile: InvestorProfile,
    user_query: str = "monitoring",
) -> MonitoringRunRequest:
    portfolio = response.final_report.portfolio
    return MonitoringRunRequest(
        profile=profile,
        active_portfolio=CandidatePortfolio(**portfolio.model_dump()),
        user_query=user_query,
    )


def build_profile_from_form(
    *,
    risk_profile: str,
    horizon_years: int,
    target: str,
    investment_amount: float | None,
    income_preference: str | None,
    sector_restrictions: list[str],
    country_restrictions: list[str],
    max_asset_weight: float,
    max_sector_weight: float,
    allowed_asset_classes: list[str],
    forbidden_assets: list[str],
    max_drawdown_tolerance: float,
    min_cash_weight: float,
    max_correlation_threshold: float,
    rebalancing_mode: str,
    period_days: int,
    drift_threshold: float,
    review_frequency: str,
) -> InvestorProfile:
    return InvestorProfile(
        risk_profile=RiskProfile(risk_profile),
        horizon_years=horizon_years,
        target=target,
        investment_amount=investment_amount,
        income_preference=income_preference or None,
        sector_restrictions=sector_restrictions,
        country_restrictions=country_restrictions,
        constraints=Constraints(
            max_asset_weight=max_asset_weight,
            max_sector_weight=max_sector_weight,
            allowed_asset_classes=allowed_asset_classes,
            forbidden_assets=forbidden_assets,
            max_drawdown_tolerance=max_drawdown_tolerance,
            min_cash_weight=min_cash_weight,
            max_correlation_threshold=max_correlation_threshold,
        ),
        rebalancing_policy=RebalancingPolicy(
            mode=rebalancing_mode,
            period_days=period_days,
            drift_threshold=drift_threshold,
            review_frequency=review_frequency,
        ),
    )
