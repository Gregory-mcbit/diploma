from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Streamlit's local source watcher aggressively inspects lazy transformer modules,
# which produces noisy torchvision-related stack traces unrelated to this app.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")

import streamlit as st

from app.api.routes import health_handler, run_monitoring_handler, run_portfolio_handler
from app.api.schemas import MonitoringRunRequest, PortfolioRunRequest, PortfolioRunResponse
from app.domain.asset_metadata import ASSET_CLASS_MAP, ASSET_COUNTRY_MAP, ASSET_SECTOR_MAP, ASSET_STYLE_MAP
from app.observability.trace_store import load_execution_traces
from app.ui.helpers import (
    asset_description,
    asset_factor_dataframe,
    asset_overview_dataframe,
    asset_display_name,
    asset_selection_summary,
    backtest_curve_dataframe,
    backtest_drawdown_dataframe,
    build_monitoring_request_from_portfolio_response,
    build_profile_from_form,
    decision_log_to_dataframe,
    feature_snapshot_dataframe,
    format_float,
    format_pct,
    fundamentals_dataframe,
    list_to_dataframe,
    mapping_to_dataframe,
    model_to_pretty_json,
    parse_json_mapping,
    selected_assets_summary_dataframe,
    split_csv_input,
    trace_log_to_dataframe,
    weights_to_dataframe,
)


st.set_page_config(
    page_title="Финансовый AI Мультиагент",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)


_EXECUTOR = ThreadPoolExecutor(max_workers=2)

PORTFOLIO_PIPELINE_STAGES = [
    "Разбираю инвестиционный запрос",
    "Собираю рыночные данные, макро и новости",
    "Считаю сигналы модели и сводный скор активов",
    "Определяю рыночный режим и ограничения",
    "Строю портфель и проверяю ограничения",
    "Прогоняю бэктест и рассчитываю риск",
    "Проверяю решение критиком",
    "Готовлю итоговый аналитический отчет",
]

MONITORING_PIPELINE_STAGES = [
    "Поднимаю текущий профиль и активный портфель",
    "Обновляю рыночные данные, макро и новости",
    "Пересчитываю сигналы и рыночный режим",
    "Проверяю риск, policy и дрейф портфеля",
    "Готовлю решение мониторинга и audit trail",
]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f8fb;
            --bg-elevated: #ffffff;
            --bg-soft: #eef3f8;
            --surface: rgba(255, 255, 255, 0.94);
            --surface-strong: rgba(255, 255, 255, 0.98);
            --surface-accent: rgba(241, 246, 252, 0.98);
            --border: rgba(148, 163, 184, 0.28);
            --border-strong: rgba(100, 116, 139, 0.26);
            --text: #111827;
            --text-soft: #4b5563;
            --text-muted: #6b7280;
            --accent: #0f766e;
            --accent-2: #2563eb;
            --accent-3: #7c3aed;
            --success: #059669;
            --warning: #d97706;
            --danger: #e11d48;
            --shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(15, 118, 110, 0.10), transparent 25%),
                radial-gradient(circle at 100% 0%, rgba(37, 99, 235, 0.10), transparent 24%),
                linear-gradient(180deg, #fbfdff 0%, #f6f8fb 45%, #eef3f8 100%);
            color: var(--text);
        }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu {
            display: none !important;
        }
        .main .block-container {
            padding-top: 1.1rem;
            max-width: 1400px;
        }
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {
            color: var(--text);
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        .stMarkdown, .stCaption, .stText, .stAlert, .stCodeBlock {
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(244, 248, 252, 0.98) 100%);
            border-right: 1px solid var(--border-strong);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        [data-testid="stSidebar"] * {
            color: var(--text);
        }
        .sidebar-shell {
            padding: 0.25rem 0 0.6rem 0;
        }
        .sidebar-status {
            background: linear-gradient(135deg, rgba(220, 252, 231, 0.96) 0%, rgba(209, 250, 229, 0.96) 100%);
            border: 1px solid rgba(5, 150, 105, 0.20);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            color: #065f46;
            font-weight: 600;
            box-shadow: 0 12px 24px rgba(6, 95, 70, 0.10);
            margin-bottom: 1rem;
        }
        .sidebar-title {
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin: 0.25rem 0 0.5rem 0;
        }
        .sidebar-copy {
            color: var(--text-soft);
            font-size: 0.97rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }
        .sidebar-section-label {
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.73rem;
            margin: 1.1rem 0 0.6rem 0;
        }
        .hero {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 22%),
                radial-gradient(circle at 85% 15%, rgba(37, 99, 235, 0.10), transparent 26%),
                linear-gradient(135deg, #ffffff 0%, #f2f7fd 55%, #eef4ff 100%);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 26px;
            padding: 1.7rem 1.9rem;
            color: var(--text);
            box-shadow: var(--shadow);
            margin-bottom: 1.4rem;
        }
        .hero h1, .hero h2, .hero h3, .hero p {
            color: var(--text);
            margin: 0;
        }
        .hero p {
            color: var(--text-soft);
            font-size: 1rem;
        }
        .section-card {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(247, 250, 253, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.15rem;
            backdrop-filter: blur(10px);
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.8rem;
            margin: 0.5rem 0 1rem 0;
        }
        .insight-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(241, 246, 252, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }
        .insight-label {
            color: var(--text-soft);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.35rem;
            line-height: 1.3;
            white-space: normal;
        }
        .insight-value {
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        .asset-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.3rem 0 1rem 0;
        }
        .asset-chip {
            background: rgba(37, 99, 235, 0.08);
            border: 1px solid rgba(37, 99, 235, 0.18);
            color: #1d4ed8;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.87rem;
        }
        .small-note {
            color: var(--text-muted);
            font-size: 0.88rem;
        }
        .metric-note {
            color: var(--text-soft);
            font-size: 0.92rem;
        }
        [data-testid="stTabs"] button {
            color: var(--text-soft);
            background: rgba(226, 232, 240, 0.72);
            border-radius: 12px 12px 0 0;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--text);
            background: rgba(255, 255, 255, 0.96);
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            background: var(--surface-strong);
        }
        [data-testid="stDataFrame"] div {
            color: var(--text);
        }
        .stAlert {
            border-radius: 14px;
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--border);
            border-radius: 16px;
        }
        .stButton > button {
            background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
            color: #111827 !important;
            border: 1px solid rgba(100, 116, 139, 0.28);
            border-radius: 14px;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
        }
        .stButton > button * {
            color: #111827 !important;
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
            border-color: rgba(37, 99, 235, 0.28);
            color: #111827 !important;
            transform: translateY(-1px);
        }
        .stButton > button:focus,
        .stButton > button:active {
            color: #111827 !important;
            border-color: rgba(37, 99, 235, 0.36);
            background: #eaf2ff;
        }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 52px;
            justify-content: flex-start;
            font-weight: 600;
            border-radius: 16px;
            box-shadow: none;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(241, 245, 249, 0.94) 100%);
            color: #111827 !important;
            border: 1px solid rgba(148, 163, 184, 0.24);
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(226, 232, 240, 0.98) 100%);
            color: #111827 !important;
        }
        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background: var(--bg-soft);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: var(--bg-soft);
            color: var(--text);
            border: 1px solid var(--border);
        }
        div[data-baseweb="base-input"] {
            background: var(--bg-soft);
        }
        .stSlider [data-baseweb="slider"] {
            padding-top: 0.7rem;
        }
        .stSlider [role="slider"] {
            background: var(--accent-2);
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14);
        }
        .stSlider div[data-testid="stTickBar"] {
            background: rgba(100, 116, 139, 0.16);
        }
        code, pre {
            background: rgba(241, 245, 249, 0.96) !important;
            color: #0f172a !important;
            border-radius: 12px !important;
        }
        .page-intro {
            text-align: center;
            padding: 0.3rem 0 1.3rem 0;
        }
        .page-kicker {
            display: inline-block;
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.16);
            color: #0f766e;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            margin-bottom: 0.85rem;
        }
        .page-intro h1 {
            font-size: clamp(2.4rem, 5vw, 4rem);
            line-height: 0.98;
            margin: 0;
            letter-spacing: -0.05em;
        }
        .page-intro p {
            max-width: 840px;
            margin: 0.9rem auto 0 auto;
            color: var(--text-soft);
            font-size: 1.05rem;
            line-height: 1.7;
        }
        .loader-shell {
            display: grid;
            place-items: center;
            min-height: 310px;
            margin: 1.2rem 0 1.6rem 0;
            text-align: center;
            border-radius: 26px;
            background:
                radial-gradient(circle at top, rgba(15, 118, 110, 0.10), transparent 28%),
                radial-gradient(circle at 80% 10%, rgba(37, 99, 235, 0.12), transparent 26%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(241, 246, 252, 0.98) 100%);
            border: 1px solid rgba(148, 163, 184, 0.26);
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        .loader-inner {
            width: min(780px, 92%);
            padding: 1.6rem 1.2rem;
        }
        .loader-ring {
            width: 84px;
            height: 84px;
            margin: 0 auto 1rem auto;
            border-radius: 50%;
            border: 2px solid rgba(148, 163, 184, 0.20);
            border-top-color: rgba(15, 118, 110, 0.95);
            border-right-color: rgba(37, 99, 235, 0.88);
            animation: loader-spin 1.1s linear infinite;
            box-shadow: 0 0 30px rgba(15, 118, 110, 0.12);
        }
        .loader-title {
            font-size: clamp(2rem, 4vw, 3.15rem);
            font-weight: 800;
            line-height: 1.02;
            letter-spacing: -0.05em;
            margin-bottom: 0.6rem;
        }
        .loader-subtitle {
            color: var(--text-soft);
            font-size: 1.04rem;
            margin-bottom: 1.35rem;
        }
        .loader-steps {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.7rem;
            text-align: left;
        }
        .loader-step {
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(248, 250, 252, 0.82);
            color: var(--text-muted);
            transition: all 180ms ease;
        }
        .loader-step strong {
            display: block;
            color: inherit;
            margin-bottom: 0.22rem;
            font-size: 0.96rem;
        }
        .loader-step span {
            font-size: 0.82rem;
            color: inherit;
        }
        .loader-step.is-done {
            color: #065f46;
            border-color: rgba(5, 150, 105, 0.18);
            background: linear-gradient(180deg, rgba(220, 252, 231, 0.88) 0%, rgba(209, 250, 229, 0.88) 100%);
        }
        .loader-step.is-active {
            color: #1d4ed8;
            border-color: rgba(37, 99, 235, 0.24);
            background: linear-gradient(180deg, rgba(239, 246, 255, 0.96) 0%, rgba(219, 234, 254, 0.96) 100%);
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.12);
        }
        @keyframes loader-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        hr {
            border-color: rgba(148, 163, 184, 0.24);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_title(title: str, subtitle: str | None = None) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def _close_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _render_hero(title: str, body: str) -> None:
    html = f'<div class="hero"><h2>{title}</h2><p style="margin-top:0.55rem;">{body}</p></div>'
    st.markdown(html, unsafe_allow_html=True)


def _render_page_intro(kicker: str, title: str, body: str) -> None:
    html = (
        '<div class="page-intro">'
        f'<div class="page-kicker">{kicker}</div>'
        f'<h1>{title}</h1>'
        f'<p>{body}</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_insight_cards(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="insight-card"><div class="insight-label">{label}</div><div class="insight-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="insight-grid">{cards}</div>', unsafe_allow_html=True)


def _render_asset_chip_row(tickers: list[str]) -> None:
    chips = "".join(f'<span class="asset-chip">{ticker}</span>' for ticker in tickers)
    st.markdown(f'<div class="asset-chip-row">{chips}</div>', unsafe_allow_html=True)


def _render_pipeline_loader(title: str, subtitle: str, stages: list[str], active_index: int) -> None:
    steps_html_parts: list[str] = []
    for index, stage in enumerate(stages):
        state_class = "is-active" if index == active_index else "is-done" if index < active_index else ""
        status = "Идет сейчас" if index == active_index else "Завершено" if index < active_index else "Следующий этап"
        steps_html_parts.append(
            f'<div class="loader-step {state_class}"><strong>{index + 1}. {stage}</strong><span>{status}</span></div>'
        )
    steps_html = "".join(steps_html_parts)
    html = (
        '<div class="loader-shell">'
        '<div class="loader-inner">'
        '<div class="loader-ring"></div>'
        f'<div class="loader-title">{title}</div>'
        f'<div class="loader-subtitle">{subtitle}</div>'
        f'<div class="loader-steps">{steps_html}</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _run_with_pipeline_loader(task: Any, *, title: str, stages: list[str], subtitle: str) -> Any:
    placeholder = st.empty()
    future = _EXECUTOR.submit(task)
    started_at = time.monotonic()
    current_index = 0

    while not future.done():
        elapsed = time.monotonic() - started_at
        next_index = min(int(elapsed / 1.35), len(stages) - 1)
        if next_index != current_index:
            current_index = next_index
        with placeholder.container():
            _render_pipeline_loader(title, subtitle, stages, current_index)
        time.sleep(0.18)

    placeholder.empty()
    return future.result()


def _render_asset_deep_dive(response: PortfolioRunResponse) -> None:
    selected_assets = response.final_report.portfolio.selected_assets
    if not selected_assets:
        st.info("В этом запуске нет активов для детализации.")
        return

    for ticker in selected_assets:
        score = response.asset_scores.get(ticker)
        fundamentals = response.fundamentals.get(ticker)
        features = response.features.get(ticker)
        news_item = response.news_digest.by_ticker.get(ticker) if response.news_digest else None
        inclusion_reason = response.final_report.inclusion_reasons.get(ticker, "Отдельная причина включения не зафиксирована.")
        weight = response.final_report.portfolio.weights.get(ticker)

        with st.expander(f"{asset_display_name(ticker)} · {ticker} · вес {format_pct(weight, 1)}", expanded=False):
            _render_insight_cards(
                [
                    ("Вес", format_pct(weight, 1)),
                    ("Общий скор", format_float(score.overall_score if score else None, 3)),
                    ("Уверенность", format_pct(score.confidence if score else None, 0)),
                    ("Класс", ASSET_CLASS_MAP.get(ticker, "н/д")),
                    ("Сектор", ASSET_SECTOR_MAP.get(ticker, "н/д")),
                    ("Стиль", ASSET_STYLE_MAP.get(ticker, "н/д")),
                ]
            )

            st.markdown("**Что это за инструмент**")
            st.write(asset_description(ticker))
            st.markdown("**Инвестиционная логика по активу**")
            st.write(inclusion_reason)
            st.markdown(
                f'<div class="small-note">Страна: {ASSET_COUNTRY_MAP.get(ticker, "н/д")} · '
                f'Новостей в digest: {news_item.article_count if news_item else 0}</div>',
                unsafe_allow_html=True,
            )

            col_factors, col_fundamentals = st.columns(2)
            with col_factors:
                st.markdown("**Факторы и скоринг**")
                factor_frame = asset_factor_dataframe(score)
                if factor_frame.empty:
                    st.info("Факторная детализация недоступна.")
                else:
                    st.dataframe(factor_frame, width="stretch", hide_index=True)
            with col_fundamentals:
                st.markdown("**Фундаментальные метрики**")
                fundamentals_frame = fundamentals_dataframe(fundamentals)
                if fundamentals_frame.empty:
                    st.info("Фундаментальные метрики не найдены.")
                else:
                    st.dataframe(fundamentals_frame, width="stretch", hide_index=True)

            st.markdown("**Фичи модели XGBoost**")
            feature_frame = feature_snapshot_dataframe(features)
            if feature_frame.empty:
                st.info("Feature snapshot для этого актива отсутствует.")
            else:
                st.dataframe(feature_frame, width="stretch", hide_index=True)

            st.markdown("**Новостной контекст**")
            if news_item:
                news_summary = {
                    "Статей": news_item.article_count,
                    "Позитивных": news_item.positive_count,
                    "Нейтральных": news_item.neutral_count,
                    "Негативных": news_item.negative_count,
                    "Источники": ", ".join(news_item.sources[:5]) if news_item.sources else "н/д",
                }
                st.dataframe(mapping_to_dataframe(news_summary, "Показатель", "Значение"), width="stretch", hide_index=True)
                if news_item.titles:
                    st.markdown("**Последние заголовки**")
                    for title in news_item.titles:
                        st.write(f"- {title}")
            else:
                st.info("В news digest нет отдельных материалов по этому активу.")


def _render_health() -> None:
    health = health_handler()
    st.sidebar.success(f"{health.service}: {health.status}")


def _render_portfolio_summary(response: PortfolioRunResponse) -> None:
    report = response.final_report
    portfolio = report.portfolio
    regime = response.market_regime
    backtest = response.backtest_result
    risk = response.risk_report
    critic = response.critic_report

    _render_hero(
        "Инвестиционная рекомендация",
        f"Режим: {regime.current_regime.value} | Вердикт критика: {critic.verdict.value} | "
        f"Запрос: {response.audit.request_id}",
    )

    _render_insight_cards(
        [
            ("Активов в портфеле", str(len(portfolio.selected_assets))),
            ("Кэш", f"{portfolio.cash_weight:.1%}"),
            ("Доходность портфеля", f"{backtest.portfolio_total_return:.1%}"),
            ("Доходность рынка", f"{backtest.benchmark_total_return:.1%}"),
            ("Волатильность", f"{risk.portfolio_volatility:.1%}"),
            ("Максимальная просадка", f"{risk.max_drawdown_estimate:.1%}"),
        ]
    )

    _render_insight_cards(
        [
            ("Доходность vs рынок", f"{format_pct(backtest.portfolio_total_return, 1)} / {format_pct(backtest.benchmark_total_return, 1)}"),
            ("Режим аллокации", portfolio.allocation_mode),
            ("Средняя корреляция", format_float(risk.avg_correlation, 2)),
            ("HHI концентрации", format_float(risk.concentration_hhi, 2)),
            ("VaR95 дневной", format_pct(risk.var_95, 2)),
            ("Уровень уверенности режима", format_pct(regime.confidence, 0)),
        ]
    )

    if portfolio.selected_assets:
        _render_asset_chip_row(portfolio.selected_assets)

    tab_overview, tab_holdings, tab_assets, tab_policy, tab_exclusions = st.tabs(
        ["Обзор", "Состав портфеля", "Детально по активам", "Policy и риск", "Включения и исключения"]
    )

    with tab_overview:
        left, right = st.columns([1.1, 0.9])
        with left:
            _card_title("Итоговая сводка")
            st.write(report.executive_summary)
            st.markdown("**Контекст режима**")
            st.write(report.regime_context)
            st.markdown("**Вердикт критика**")
            st.write(critic.recommended_action)
            if critic.issues:
                st.markdown("**Что проверял критик**")
                for issue in critic.issues:
                    st.write(f"- {issue}")
            if report.process_summary:
                st.markdown("**Как проходили пересборки**")
                for line in report.process_summary:
                    st.write(f"- {line}")
            _close_card()

            _card_title("Почему именно эти активы")
            for ticker in portfolio.selected_assets:
                st.markdown(f"**{asset_display_name(ticker)} ({ticker})**")
                st.write(asset_selection_summary(response, ticker))
            _close_card()

            _card_title("Конструкция портфеля")
            if portfolio.rationale:
                st.markdown("**Обоснование**")
                for sentence in portfolio.rationale:
                    st.write(f"- {sentence}")
            if portfolio.construction_notes:
                st.markdown("**Заметки по конструкции**")
                for note in portfolio.construction_notes:
                    st.write(f"- {note}")
            if report.uncertainty_notes:
                st.markdown("**Неопределенность**")
                for note in report.uncertainty_notes:
                    st.write(f"- {note}")
            st.markdown("**Дисклеймер по рискам**")
            st.write(report.risk_disclaimer)
            _close_card()

        with right:
            _card_title("Снимок бэктеста")
            curve_frame = backtest_curve_dataframe(backtest)
            if curve_frame.empty:
                st.info("Кривая бэктеста недоступна для этого запуска.")
            else:
                st.markdown("**Кривая доходности: портфель vs SPY**")
                st.line_chart(curve_frame, width="stretch")
                st.caption("Значения на графике показаны как накопленная доходность, % от начала окна бэктеста.")
            drawdown_frame = backtest_drawdown_dataframe(backtest)
            if not drawdown_frame.empty:
                st.markdown("**Скользящая просадка портфеля**")
                st.line_chart(drawdown_frame, width="stretch")
            st.dataframe(
                mapping_to_dataframe(
                    {
                        "Суммарная доходность портфеля": f"{backtest.portfolio_total_return:.2%}",
                        "Суммарная доходность бенчмарка": f"{backtest.benchmark_total_return:.2%}",
                        "Доходность equal weight": f"{backtest.equal_weight_total_return:.2%}",
                        "Геометрическая доходность портфеля": f"{backtest.portfolio_geometric_mean_return:.2%}",
                        "Волатильность": f"{backtest.portfolio_volatility:.2%}",
                        "Макс. просадка": f"{backtest.portfolio_max_drawdown:.2%}",
                        "Оборот": f"{backtest.turnover:.2%}",
                        "Наблюдений": backtest.observations,
                    },
                    "Показатель",
                    "Значение",
                ),
                width="stretch",
                hide_index=True,
            )
            if report.backtest_summary:
                st.markdown("**Что именно проверял бэктест**")
                for line in report.backtest_summary:
                    st.write(f"- {line}")
            _close_card()

            _card_title("Карта выбранных позиций")
            selected_summary_frame = selected_assets_summary_dataframe(response)
            if selected_summary_frame.empty:
                st.info("Выбранные позиции отсутствуют.")
            else:
                st.dataframe(selected_summary_frame, width="stretch", hide_index=True)
            _close_card()

    with tab_holdings:
        _card_title("Текущие веса и аналитика по позициям", "Отсюда удобно быстро понять, что именно сейчас держит портфель.")
        weights_frame = weights_to_dataframe(portfolio.weights)
        if not weights_frame.empty:
            st.bar_chart(weights_frame.set_index("Ticker"))
        overview_frame = asset_overview_dataframe(response)
        if overview_frame.empty:
            st.info("Детализация по активам недоступна.")
        else:
            st.dataframe(overview_frame, width="stretch", hide_index=True)
        _close_card()

    with tab_assets:
        _card_title("Deep Dive по каждому активу", "Для каждой позиции показываются факторы скоринга, фундаментал, model features и news digest.")
        _render_asset_deep_dive(response)
        _close_card()

    with tab_policy:
        _card_title("Policy и риск")
        policy = response.effective_policy
        st.markdown("**Примененные policy rules**")
        for summary in policy.applied_rule_summaries:
            st.write(f"- {summary}")
        if report.policy_summary:
            st.markdown("**Итог policy для клиента**")
            for summary in report.policy_summary:
                st.write(f"- {summary}")
        st.markdown("**Предупреждения по риску**")
        st.dataframe(
            mapping_to_dataframe(
                {
                    "Волатильность": format_pct(risk.portfolio_volatility, 2),
                    "Максимальная просадка": format_pct(risk.max_drawdown_estimate, 2),
                    "Средняя корреляция": format_float(risk.avg_correlation, 3),
                    "HHI концентрации": format_float(risk.concentration_hhi, 3),
                    "VaR95 дневной": format_pct(risk.var_95, 2),
                },
                "Метрика",
                "Значение",
            ),
            width="stretch",
            hide_index=True,
        )
        if risk.warnings:
            for warning in risk.warnings:
                st.write(f"- {warning}")
        else:
            st.write("Мягких предупреждений нет.")
        st.markdown("**Жесткие нарушения**")
        if risk.violations:
            for violation in risk.violations:
                st.error(violation)
        else:
            st.success("Жестких нарушений policy или риска нет.")
        _close_card()

    with tab_exclusions:
        _card_title("Включения и исключения")
        if report.inclusion_reasons:
            st.markdown("**Включено**")
            st.dataframe(
                mapping_to_dataframe(report.inclusion_reasons, "Тикер", "Причина"),
                width="stretch",
                hide_index=True,
            )
        if report.exclusion_reasons:
            st.markdown("**Исключено**")
            st.dataframe(
                mapping_to_dataframe(report.exclusion_reasons, "Тикер", "Причина"),
                width="stretch",
                hide_index=True,
            )
        if response.news_digest:
            st.markdown("**Сводка news digest**")
            st.dataframe(
                mapping_to_dataframe(
                    {
                        "Всего статей": response.news_digest.article_count,
                        "Покрыто тикеров": len(response.news_digest.tickers),
                        "Позитивных": response.news_digest.sentiment_totals.get("positive", 0),
                        "Нейтральных": response.news_digest.sentiment_totals.get("neutral", 0),
                        "Негативных": response.news_digest.sentiment_totals.get("negative", 0),
                    },
                    "Показатель",
                    "Значение",
                ),
                width="stretch",
                hide_index=True,
            )
        _close_card()


def _render_monitoring_summary(response: Any) -> None:
    decision = response.monitoring_decision
    regime = response.market_regime
    risk = response.risk_report

    _render_hero(
        "Решение мониторинга",
        f"Действие: {decision.action.value} | Режим: {regime.current_regime.value} | "
        f"Запрос: {response.audit.request_id}",
    )
    _render_insight_cards(
        [
            ("Действие", decision.action.value),
            ("Триггеры", str(len(decision.trigger_flags))),
            ("Нарушения", str(len(risk.violations))),
            ("Предупреждения", str(len(risk.warnings))),
        ]
    )

    left, right = st.columns([1.2, 1.0])
    with left:
        _card_title("Решение")
        st.write(decision.summary)
        if decision.reasons:
            st.markdown("**Причины**")
            for reason in decision.reasons:
                st.write(f"- {reason}")
        if decision.trigger_flags:
            st.markdown("**Триггер-флаги**")
            st.dataframe(list_to_dataframe(decision.trigger_flags, "Триггер"), width="stretch", hide_index=True)
        _close_card()

    with right:
        _card_title("Риск и policy")
        if risk.violations:
            for violation in risk.violations:
                st.error(violation)
        else:
            st.success("Жестких нарушений нет.")
        if risk.warnings:
            for warning in risk.warnings:
                st.warning(warning)
        st.markdown("**Сводка policy**")
        for summary in response.effective_policy.applied_rule_summaries:
            st.write(f"- {summary}")
        _close_card()


def _render_audit(audit: Any) -> None:
    tab_overview, tab_trace, tab_provenance, tab_memory, tab_raw = st.tabs(
        ["Обзор", "Trace Log", "Provenance", "Memory Refs", "Raw JSON"]
    )
    with tab_overview:
        st.dataframe(
            mapping_to_dataframe(
                {
                    "Request ID": audit.request_id,
                    "Correlation ID": audit.correlation_id,
                    "Записей decision log": len(audit.decision_log),
                    "Trace-событий": len(audit.trace_log),
                    "Memory refs": len(audit.memory_refs),
                }
            ),
            width="stretch",
            hide_index=True,
        )
        if audit.freshness_map:
            st.markdown("**Карта свежести**")
            st.dataframe(
                mapping_to_dataframe(audit.freshness_map, "Артефакт", "Свежесть"),
                width="stretch",
                hide_index=True,
            )
        if audit.decision_log:
            st.markdown("**Decision Log**")
            st.dataframe(
                decision_log_to_dataframe(audit.decision_log),
                width="stretch",
                hide_index=True,
            )
    with tab_trace:
        st.dataframe(
            trace_log_to_dataframe(audit.trace_log),
            width="stretch",
            hide_index=True,
        )
    with tab_provenance:
        for artifact, record in audit.provenance.items():
            with st.expander(artifact, expanded=False):
                st.code(model_to_pretty_json(record), language="json")
    with tab_memory:
        if audit.memory_refs:
            st.dataframe(
                mapping_to_dataframe(
                    {index: ref for index, ref in enumerate(audit.memory_refs)},
                    "Индекс",
                    "MemoryRef",
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Для этого запуска memory references не зафиксированы.")
    with tab_raw:
        st.code(model_to_pretty_json(audit), language="json")


def _build_profile_form(prefix: str = "monitor") -> Any:
    col_a, col_b = st.columns(2)
    with col_a:
        risk_profile = st.selectbox(
            "Риск-профиль",
            ["moderate", "conservative", "aggressive"],
            key=f"{prefix}_risk_profile",
        )
        horizon_years = st.number_input("Горизонт (лет)", min_value=1, max_value=50, value=10, key=f"{prefix}_horizon")
        target = st.text_input("Цель", value="Сбалансированный долгосрочный рост", key=f"{prefix}_target")
        investment_amount = st.number_input(
            "Сумма инвестиций",
            min_value=0.0,
            value=250000.0,
            step=10000.0,
            key=f"{prefix}_investment_amount",
        )
        income_preference = st.selectbox(
            "Предпочтение по доходности",
            ["", "income", "growth"],
            key=f"{prefix}_income_preference",
        )
        sector_restrictions = split_csv_input(
            st.text_input("Ограничения по секторам (CSV)", value="", key=f"{prefix}_sector_restrictions")
        )
        country_restrictions = split_csv_input(
            st.text_input("Ограничения по странам (CSV)", value="", key=f"{prefix}_country_restrictions")
        )
    with col_b:
        max_asset_weight = st.slider("Макс. вес актива", 0.01, 1.0, 0.25, 0.01, key=f"{prefix}_max_asset")
        max_sector_weight = st.slider("Макс. вес сектора", 0.01, 1.0, 0.35, 0.01, key=f"{prefix}_max_sector")
        max_drawdown_tolerance = st.slider("Макс. допустимая просадка", 0.01, 1.0, 0.20, 0.01, key=f"{prefix}_drawdown")
        min_cash_weight = st.slider("Мин. доля кэша", 0.0, 0.5, 0.02, 0.01, key=f"{prefix}_min_cash")
        max_correlation_threshold = st.slider("Макс. порог корреляции", 0.1, 1.0, 0.85, 0.01, key=f"{prefix}_corr")
        allowed_asset_classes = split_csv_input(
            st.text_input(
                "Разрешенные классы активов (CSV)",
                value="stocks,bonds,commodities",
                key=f"{prefix}_asset_classes",
            )
        )
        forbidden_assets = split_csv_input(
            st.text_input("Запрещенные активы (CSV)", value="", key=f"{prefix}_forbidden_assets")
        )
        rebalancing_mode = st.selectbox(
            "Режим ребалансировки",
            ["threshold_and_periodic", "rank_allocation"],
            key=f"{prefix}_rebalance_mode",
        )
        period_days = st.number_input("Период пересмотра (дней)", min_value=1, max_value=365, value=30, key=f"{prefix}_period_days")
        drift_threshold = st.slider("Порог дрейфа", 0.0, 0.5, 0.05, 0.01, key=f"{prefix}_drift")
        review_frequency = st.selectbox(
            "Частота пересмотра",
            ["monthly", "weekly", "quarterly"],
            key=f"{prefix}_review_frequency",
        )

    return build_profile_from_form(
        risk_profile=risk_profile,
        horizon_years=int(horizon_years),
        target=target,
        investment_amount=float(investment_amount) if investment_amount else None,
        income_preference=income_preference or None,
        sector_restrictions=sector_restrictions,
        country_restrictions=country_restrictions,
        max_asset_weight=float(max_asset_weight),
        max_sector_weight=float(max_sector_weight),
        allowed_asset_classes=allowed_asset_classes,
        forbidden_assets=forbidden_assets,
        max_drawdown_tolerance=float(max_drawdown_tolerance),
        min_cash_weight=float(min_cash_weight),
        max_correlation_threshold=float(max_correlation_threshold),
        rebalancing_mode=rebalancing_mode,
        period_days=int(period_days),
        drift_threshold=float(drift_threshold),
        review_frequency=review_frequency,
    )


def _portfolio_page() -> None:
    _render_page_intro(
        "FINANCE AI CONTROL",
        "Запуск инвестиционного портфеля",
        "Запусти полный инвестиционный пайплайн: профиль, данные, сигналы модели, режим рынка, policy, портфель, бэктест, риск, критик и итоговый отчет.",
    )

    default_query = (
        "Собери умеренный долгосрочный портфель с жестким контролем риска, сохрани небольшой кэш-буфер "
        "и подробно объясни рекомендацию."
    )
    query = st.text_area("Инвестиционный запрос", value=default_query, height=140)

    left_spacer, center_action, right_spacer = st.columns([0.36, 0.28, 0.36])
    with center_action:
        run_clicked = st.button("Запустить портфель", width="stretch", type="primary")
   
    if run_clicked:
        try:
            response = _run_with_pipeline_loader(
                lambda: run_portfolio_handler(PortfolioRunRequest(user_query=query)),
                title="Собираю инвестиционную рекомендацию",
                subtitle="Шаги ниже идут по реальному порядку пайплайна, но описаны простым человеческим языком.",
                stages=PORTFOLIO_PIPELINE_STAGES,
            )
            st.session_state["portfolio_response"] = response
            st.success("Портфельный граф завершен. Ниже уже готова полная аналитика по результату.")
        except Exception as exc:
            st.session_state.pop("portfolio_response", None)
            st.error(str(exc))
            with st.expander("Технические детали"):
                st.exception(exc)

    response = st.session_state.get("portfolio_response")
    if response:
        _render_portfolio_summary(response)
        st.markdown("---")
        st.subheader("Аудит")
        _render_audit(response.audit)


def _build_manual_monitoring_request() -> MonitoringRunRequest:
    profile = _build_profile_form(prefix="manual_monitor")
    weights_text = st.text_area(
        "Веса активного портфеля (JSON)",
        value='{"SPY": 0.45, "TLT": 0.30, "GLD": 0.15}',
        height=140,
    )
    weights = parse_json_mapping(weights_text, "Active Portfolio Weights")
    selected_assets = list(weights.keys())
    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    rationale_text = st.text_area(
        "Рационали портфеля (по одной строке)",
        value="Существующий сбалансированный портфель.\nЗащитный слой уже присутствует.\nЗапрошен мониторинговый прогон.",
        height=100,
    )
    rationale = [line.strip() for line in rationale_text.splitlines() if line.strip()]
    user_query = st.text_input("Запрос для мониторинга", value="monitoring")
    return MonitoringRunRequest(
        profile=profile,
        active_portfolio={
            "selected_assets": selected_assets,
            "weights": weights,
            "cash_weight": cash_weight,
            "rationale": rationale or ["Существующий портфель."],
        },
        user_query=user_query,
    )


def _monitoring_page() -> None:
    _render_page_intro(
        "Monitoring graph",
        "Запуск мониторинга портфеля",
        "Проверь существующий портфель на drift, режим рынка, риск и policy. Можно запустить мониторинг от последней рекомендации или вручную задать портфель и профиль.",
    )

    source_mode = st.radio(
        "Источник мониторинга",
        ["Использовать последний запуск портфеля", "Ручной ввод"],
        horizontal=True,
    )

    request: MonitoringRunRequest | None = None
    if source_mode == "Использовать последний запуск портфеля":
        portfolio_response: PortfolioRunResponse | None = st.session_state.get("portfolio_response")
        if not portfolio_response:
            st.info("Сначала запусти портфель или переключись на ручной ввод.")
        else:
            st.write("Найден последний результат портфеля. Теперь укажи профиль инвестора для мониторинга.")
            profile = _build_profile_form(prefix="portfolio_monitor")
            request = build_monitoring_request_from_portfolio_response(
                portfolio_response,
                profile=profile,
                user_query="monitoring",
            )
    else:
        request = _build_manual_monitoring_request()

    if st.button("Запустить мониторинг", width="stretch", type="primary", disabled=request is None):
        try:
            response = _run_with_pipeline_loader(
                lambda: run_monitoring_handler(request),
                title="Проверяю портфель в мониторинге",
                subtitle="Этапы отражают реальную последовательность monitoring graph и дают понятное состояние ожидания.",
                stages=MONITORING_PIPELINE_STAGES,
            )
            st.session_state["monitoring_response"] = response
            st.success("Мониторинговый прогон завершен. Ниже доступно решение и audit trail.")
        except Exception as exc:
            st.exception(exc)

    response = st.session_state.get("monitoring_response")
    if response:
        _render_monitoring_summary(response)
        st.markdown("---")
        st.subheader("Аудит")
        _render_audit(response.audit)


def _trace_explorer_page() -> None:
    _render_page_intro(
        "Execution traces",
        "Журнал выполнений",
        "Просматривай сохраненные трассы выполнения, decision log и технический след каждого запуска без необходимости лезть в raw JSON вручную.",
    )

    traces = load_execution_traces()
    if not traces:
        st.info("Сохраненных execution traces пока нет.")
        return

    summary_rows = []
    for record in traces:
        summary_rows.append(
            {
                "request_id": record.get("request_id"),
                "correlation_id": record.get("correlation_id"),
                "run_type": record.get("run_type"),
                "status": record.get("status"),
                "started_at": record.get("started_at"),
                "completed_at": record.get("completed_at"),
                "trace_events": len(record.get("trace_log", [])),
                "decision_events": len(record.get("decision_log", [])),
            }
        )

    import pandas as pd

    frame = pd.DataFrame(summary_rows)
    st.dataframe(frame, width="stretch", hide_index=True)

    request_ids = [row["request_id"] for row in summary_rows]
    selected_request_id = st.selectbox("Выбери выполнение", request_ids, index=len(request_ids) - 1)
    selected_record = next(record for record in traces if record.get("request_id") == selected_request_id)

    left, right = st.columns([1.0, 1.2])
    with left:
        _card_title("Метаданные выполнения")
        st.code(model_to_pretty_json(selected_record), language="json")
        _close_card()
    with right:
        _card_title("Лента выполнения")
        st.dataframe(
            trace_log_to_dataframe(selected_record.get("trace_log", [])),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Decision Log**")
        st.dataframe(
            decision_log_to_dataframe(selected_record.get("decision_log", [])),
            width="stretch",
            hide_index=True,
        )
        _close_card()


def main() -> None:
    _inject_styles()
    _render_health()
    st.sidebar.markdown('<div class="sidebar-shell">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-title">Finance AI<br/>Control</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-label">Навигация</div>', unsafe_allow_html=True)

    pages = ["Портфель", "Мониторинг", "Журнал запусков"]
    if "ui_page" not in st.session_state:
        st.session_state["ui_page"] = "Портфель"
    for page_name in pages:
        clicked = st.sidebar.button(
            page_name,
            key=f"sidebar_nav_{page_name}",
            width="stretch",
            type="primary" if st.session_state["ui_page"] == page_name else "secondary",
        )
        if clicked and st.session_state["ui_page"] != page_name:
            st.session_state["ui_page"] = page_name
            st.rerun()

    st.sidebar.markdown('<div class="sidebar-section-label">Что внутри</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-copy">Портфель: полный инвестиционный пайплайн.<br/>Мониторинг: контроль drift и пересмотра.<br/>Журнал: история execution trace и decision log.</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    page = st.session_state["ui_page"]

    if page == "Портфель":
        _portfolio_page()
    elif page == "Мониторинг":
        _monitoring_page()
    else:
        _trace_explorer_page()


if __name__ == "__main__":
    main()
