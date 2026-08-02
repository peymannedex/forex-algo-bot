from __future__ import annotations

from datetime import timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyConfig
from fxbot.strategy.trend_following import TrendFollowingConfig, TrendFollowingStrategy


def _settings() -> TrendFollowingConfig:
    return TrendFollowingConfig(
        strategy=StrategyConfig(
            strategy_id="trend",
            primary_timeframe=Timeframe.M5,
            required_timeframes=(Timeframe.H1,),
            warmup_bars=20,
            max_data_age=timedelta(hours=2),
        )
    )


def test_trend_buy_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1020],
        higher_closes=[1.1000] * 30,
        latest_open=1.1010,
        latest_low=1.1008,
        latest_high=1.1024,
    )
    snapshot = make_snapshot(
        context,
        fast=1.1015,
        slow=1.1000,
        momentum=0.002,
        rsi=60.0,
    )
    monkeypatch.setattr(
        "fxbot.strategy.trend_following.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    strategy = TrendFollowingStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    )
    decision = strategy.evaluate(context)
    assert decision.action is SignalAction.BUY
    assert decision.stop_loss is not None and decision.stop_loss < decision.entry_price
    assert decision.take_profit is not None and decision.take_profit > decision.entry_price


def test_trend_sell_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1020] * 29 + [1.1000],
        higher_closes=[1.1020] * 30,
        latest_open=1.1010,
        latest_low=1.0996,
        latest_high=1.1012,
    )
    snapshot = make_snapshot(
        context,
        fast=1.1005,
        slow=1.1020,
        momentum=-0.002,
        rsi=40.0,
    )
    monkeypatch.setattr(
        "fxbot.strategy.trend_following.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    strategy = TrendFollowingStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_DOWN),
    )
    decision = strategy.evaluate(context)
    assert decision.action is SignalAction.SELL
    assert decision.take_profit is not None and decision.take_profit < decision.entry_price
    assert decision.stop_loss is not None and decision.stop_loss > decision.entry_price


def test_trend_holds_without_pullback_confirmation(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1050],
        higher_closes=[1.1000] * 30,
        latest_open=1.1040,
        latest_low=1.1040,
        latest_high=1.1055,
    )
    snapshot = make_snapshot(context, fast=1.101, slow=1.100, momentum=0.002, rsi=60)
    monkeypatch.setattr(
        "fxbot.strategy.trend_following.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    strategy = TrendFollowingStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    )
    decision = strategy.evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("trend_pullback_confirmation_missing",)


def test_trend_rejects_wrong_higher_timeframe_direction(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1020],
        higher_closes=[1.1000] * 30,
        latest_open=1.1010,
        latest_low=1.1008,
    )
    snapshot = make_snapshot(context, fast=1.1015, slow=1.100, momentum=0.002, rsi=60)
    monkeypatch.setattr(
        "fxbot.strategy.trend_following.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    classifier = classifier_factory(
        MarketRegime.TRENDING_UP,
        dominant=MarketRegime.TRENDING_DOWN,
    )
    decision = TrendFollowingStrategy(
        _settings(),
        regime_classifier=classifier,
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("trend_market_filter_rejected",)


def test_trend_filter_rejects_wide_spread(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1020],
        higher_closes=[1.1000] * 30,
        latest_open=1.1010,
        latest_low=1.1008,
    )
    snapshot = make_snapshot(
        context,
        fast=1.1015,
        slow=1.100,
        momentum=0.002,
        rsi=60,
        spread_to_atr=0.5,
    )
    monkeypatch.setattr(
        "fxbot.strategy.trend_following.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = TrendFollowingStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert ("filter_0", "spread_to_atr_exceeds_limit") in decision.metadata


def test_trend_config_requires_higher_timeframe() -> None:
    with pytest.raises(ValueError, match="higher timeframe"):
        TrendFollowingConfig(
            strategy=StrategyConfig(
                strategy_id="trend",
                primary_timeframe=Timeframe.M5,
            )
        )


def test_trend_returns_hold_for_short_series(make_context) -> None:
    context = make_context([1.10])
    settings = TrendFollowingConfig(
        strategy=StrategyConfig(
            strategy_id="trend",
            primary_timeframe=Timeframe.M5,
            required_timeframes=(Timeframe.H1,),
        )
    )
    decision = TrendFollowingStrategy(settings).evaluate(context)
    assert decision.action is SignalAction.HOLD
