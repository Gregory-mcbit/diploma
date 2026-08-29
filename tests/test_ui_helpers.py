from app.api.schemas import AuditPayload, PortfolioRunResponse
from app.domain.schemas import (
    AssetScore,
    BacktestResult,
    CandidatePortfolio,
    Constraints,
    CriticReport,
    DecisionLogEntry,
    CriticVerdict,
    EffectivePolicy,
    FinalRecommendation,
    FundamentalSnapshot,
    InvestorProfile,
    RegimeReport,
    RegimeType,
    RebalancingPolicy,
    RiskProfile,
    RiskReport,
    FactorScores,
    FeatureSnapshot,
    NewsDigest,
    NewsDigestItem,
    TraceEvent,
)
from app.ui.helpers import (
    asset_description,
    asset_display_name,
    asset_overview_dataframe,
    backtest_curve_dataframe,
    backtest_drawdown_dataframe,
    selected_assets_summary_dataframe,
    build_monitoring_request_from_portfolio_response,
    build_profile_from_form,
    decision_log_to_dataframe,
    mapping_to_dataframe,
    parse_json_mapping,
    split_csv_input,
    trace_log_to_dataframe,
    weights_to_dataframe,
)


def test_split_csv_input_and_parse_json_mapping():
    assert split_csv_input("us, technology,  ") == ["us", "technology"]
    assert parse_json_mapping('{"SPY": 0.5, "TLT": 0.3}', "weights") == {"SPY": 0.5, "TLT": 0.3}


def test_backtest_chart_dataframes():
    backtest = BacktestResult(
        portfolio_total_return=0.1,
        benchmark_total_return=0.08,
        equal_weight_total_return=0.07,
        portfolio_volatility=0.12,
        portfolio_max_drawdown=0.08,
        turnover=0.1,
        observations=2,
        curve_dates=["2024-01-02", "2024-01-03"],
        portfolio_curve=[1.0, 1.1],
        benchmark_curve=[1.0, 1.08],
        equal_weight_curve=[1.0, 1.07],
        drawdown_curve=[0.0, -0.02],
    )

    curve = backtest_curve_dataframe(backtest)
    drawdown = backtest_drawdown_dataframe(backtest)

    assert list(curve.columns) == ["Портфель", "SPY benchmark", "Equal weight"]
    assert round(curve.iloc[-1]["Портфель"], 2) == 10.0
    assert drawdown.iloc[-1]["Просадка портфеля"] == -2.0


def test_build_profile_from_form():
    profile = build_profile_from_form(
        risk_profile="moderate",
        horizon_years=10,
        target="Balanced growth",
        investment_amount=100000.0,
        income_preference="income",
        sector_restrictions=["technology"],
        country_restrictions=["us"],
        max_asset_weight=0.25,
        max_sector_weight=0.35,
        allowed_asset_classes=["stocks", "bonds"],
        forbidden_assets=["QQQ"],
        max_drawdown_tolerance=0.20,
        min_cash_weight=0.02,
        max_correlation_threshold=0.85,
        rebalancing_mode="threshold_and_periodic",
        period_days=30,
        drift_threshold=0.05,
        review_frequency="monthly",
    )

    assert profile.risk_profile == RiskProfile.moderate
    assert profile.income_preference == "income"
    assert profile.sector_restrictions == ["technology"]
    assert profile.constraints.forbidden_assets == ["QQQ"]
    assert profile.rebalancing_policy == RebalancingPolicy(
        mode="threshold_and_periodic",
        period_days=30,
        drift_threshold=0.05,
        review_frequency="monthly",
    )


def test_build_monitoring_request_from_portfolio_response():
    profile = InvestorProfile(
        risk_profile=RiskProfile.moderate,
        horizon_years=10,
        target="Balanced growth",
        constraints=Constraints(
            max_asset_weight=0.25,
            max_sector_weight=0.35,
            allowed_asset_classes=["stocks", "bonds", "commodities"],
            forbidden_assets=[],
            max_drawdown_tolerance=0.20,
            min_cash_weight=0.02,
            max_correlation_threshold=0.85,
        ),
    )
    response = PortfolioRunResponse(
        final_report=FinalRecommendation(
            portfolio=CandidatePortfolio(
                selected_assets=["SPY", "TLT"],
                weights={"SPY": 0.6, "TLT": 0.3},
                cash_weight=0.1,
                rationale=["Balanced growth."],
            ),
            executive_summary="Summary",
            regime_context="Context",
            risk_disclaimer="Disclaimer",
        ),
        critic_report=CriticReport(
            verdict=CriticVerdict.approve,
            issues=[],
            recommended_action="Approved.",
        ),
        market_regime=RegimeReport(
            current_regime=RegimeType.sideways,
            confidence=0.7,
            drivers=["Mixed macro", "Stable credit", "Neutral momentum"],
            is_risk_off=False,
        ),
        backtest_result=BacktestResult(
            portfolio_total_return=0.1,
            benchmark_total_return=0.08,
            equal_weight_total_return=0.07,
            portfolio_volatility=0.12,
            portfolio_max_drawdown=0.08,
            turnover=0.1,
            observations=252,
        ),
        risk_report=RiskReport(
            portfolio_volatility=0.12,
            max_drawdown_estimate=0.08,
            avg_correlation=0.3,
            violations=[],
            warnings=[],
            fit_to_profile="acceptable",
        ),
        effective_policy=EffectivePolicy(
            constraints=profile.constraints,
        ),
        audit=AuditPayload(
            request_id="req-test",
            correlation_id="corr-test",
        ),
    )

    request = build_monitoring_request_from_portfolio_response(response, profile, user_query="monitoring")

    assert request.profile == profile
    assert request.active_portfolio.selected_assets == ["SPY", "TLT"]
    assert request.user_query == "monitoring"


def test_weights_to_dataframe_sorts_descending():
    frame = weights_to_dataframe({"TLT": 0.3, "SPY": 0.6})

    assert list(frame["Ticker"]) == ["SPY", "TLT"]


def test_mapping_to_dataframe_is_arrow_safe_for_mixed_values():
    frame = mapping_to_dataframe({"Observations": 252, "Warnings": ["a", "b"], "Meta": {"x": 1}})

    assert list(frame.columns) == ["Key", "Value"]
    assert frame["Value"].dtype == object
    assert all(value is None or isinstance(value, str) for value in frame["Value"])


def test_trace_and_decision_log_frames_serialize_complex_columns():
    trace_frame = trace_log_to_dataframe(
        [
            TraceEvent(
                request_id="req-1",
                correlation_id="corr-1",
                stage="graph",
                event="execution_started",
                message="Started.",
                timestamp="2026-04-01T00:00:00Z",
                metadata={"attempt": 1, "tags": ["ui", "graph"]},
            )
        ]
    )
    decision_frame = decision_log_to_dataframe(
        [
            DecisionLogEntry(
                stage="critic",
                event="portfolio_reviewed",
                message="Reviewed.",
                timestamp="2026-04-01T00:00:00Z",
                tool_calls=["retrieve_past_mistakes"],
                rule_ids=["rule-1"],
                metadata={"verdict": "approve"},
            )
        ]
    )

    assert isinstance(trace_frame.loc[0, "metadata"], str)
    assert isinstance(decision_frame.loc[0, "tool_calls"], str)
    assert isinstance(decision_frame.loc[0, "rule_ids"], str)
    assert isinstance(decision_frame.loc[0, "metadata"], str)


def test_asset_overview_dataframe_contains_selected_assets():
    response = PortfolioRunResponse(
        final_report=FinalRecommendation(
            portfolio=CandidatePortfolio(
                selected_assets=["SPY"],
                weights={"SPY": 0.6},
                cash_weight=0.4,
                rationale=["Balanced growth."],
            ),
            executive_summary="Summary",
            regime_context="Context",
            risk_disclaimer="Disclaimer",
        ),
        critic_report=CriticReport(
            verdict=CriticVerdict.approve,
            issues=[],
            recommended_action="Approved.",
        ),
        market_regime=RegimeReport(
            current_regime=RegimeType.sideways,
            confidence=0.7,
            drivers=["Mixed macro", "Stable credit", "Neutral momentum"],
            is_risk_off=False,
        ),
        backtest_result=BacktestResult(
            portfolio_total_return=0.1,
            benchmark_total_return=0.08,
            equal_weight_total_return=0.07,
            portfolio_volatility=0.12,
            portfolio_max_drawdown=0.08,
            turnover=0.1,
            observations=252,
        ),
        risk_report=RiskReport(
            portfolio_volatility=0.12,
            max_drawdown_estimate=0.08,
            avg_correlation=0.3,
            violations=[],
            warnings=[],
            fit_to_profile="acceptable",
        ),
        effective_policy=EffectivePolicy(
            constraints=Constraints(
                max_asset_weight=0.25,
                max_sector_weight=0.35,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.20,
                min_cash_weight=0.02,
                max_correlation_threshold=0.85,
            ),
        ),
        asset_scores={
            "SPY": AssetScore(
                asset_ticker="SPY",
                factors=FactorScores(momentum=0.2, volatility=-0.1, quality=0.3),
                overall_score=0.051,
                confidence=0.84,
            )
        },
        fundamentals={
            "SPY": FundamentalSnapshot(
                ticker="SPY",
                metrics={"pe": 21.1, "roe": 0.16},
                source="test",
            )
        },
        features={
            "SPY": FeatureSnapshot(
                ticker="SPY",
                values={"mom_1m": 0.02, "vol_20d": 0.15},
            )
        },
        news_digest=NewsDigest(
            article_count=1,
            tickers=["SPY"],
            sentiment_totals={"positive": 1},
            by_ticker={
                "SPY": NewsDigestItem(
                    ticker="SPY",
                    article_count=1,
                    positive_count=1,
                    titles=["Positive macro backdrop"],
                    sources=["Reuters"],
                )
            },
        ),
        audit=AuditPayload(
            request_id="req-test",
            correlation_id="corr-test",
        ),
    )

    frame = asset_overview_dataframe(response)
    assert list(frame["Тикер"]) == ["SPY"]
    assert frame.loc[0, "Вес"] == "60.0%"
    assert frame.loc[0, "Название"] == "SPDR S&P 500 ETF"
    assert "Почему выбрано" in frame.columns


def test_asset_description_and_summary_dataframe_are_human_readable():
    response = PortfolioRunResponse(
        final_report=FinalRecommendation(
            portfolio=CandidatePortfolio(
                selected_assets=["SPY"],
                weights={"SPY": 0.6},
                cash_weight=0.4,
                rationale=["Balanced growth."],
            ),
            executive_summary="Summary",
            regime_context="Context",
            risk_disclaimer="Disclaimer",
            inclusion_reasons={"SPY": "Широкий рынок США дает базовое ядро портфеля."},
        ),
        critic_report=CriticReport(
            verdict=CriticVerdict.approve,
            issues=[],
            recommended_action="Approved.",
        ),
        market_regime=RegimeReport(
            current_regime=RegimeType.sideways,
            confidence=0.7,
            drivers=["Mixed macro", "Stable credit", "Neutral momentum"],
            is_risk_off=False,
        ),
        backtest_result=BacktestResult(
            portfolio_total_return=0.1,
            benchmark_total_return=0.08,
            equal_weight_total_return=0.07,
            portfolio_volatility=0.12,
            portfolio_max_drawdown=0.08,
            turnover=0.1,
            observations=252,
        ),
        risk_report=RiskReport(
            portfolio_volatility=0.12,
            max_drawdown_estimate=0.08,
            avg_correlation=0.3,
            violations=[],
            warnings=[],
            fit_to_profile="acceptable",
        ),
        effective_policy=EffectivePolicy(
            constraints=Constraints(
                max_asset_weight=0.25,
                max_sector_weight=0.35,
                allowed_asset_classes=["stocks", "bonds", "commodities"],
                forbidden_assets=[],
                max_drawdown_tolerance=0.20,
                min_cash_weight=0.02,
                max_correlation_threshold=0.85,
            ),
        ),
        audit=AuditPayload(
            request_id="req-test",
            correlation_id="corr-test",
        ),
    )

    assert asset_display_name("SPY") == "SPDR S&P 500 ETF"
    assert "широкий рынок акций сша" in asset_description("SPY").lower()

    frame = selected_assets_summary_dataframe(response)
    assert frame.loc[0, "Название"] == "SPDR S&P 500 ETF"
    assert frame.loc[0, "Почему выбрано"] == "Широкий рынок США дает базовое ядро портфеля."
