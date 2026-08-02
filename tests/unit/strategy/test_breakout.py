from __future__ import annotations

from datetime import timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.breakout import BreakoutStrategy, BreakoutStrategyConfig
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyConfig


def _settings() -> BreakoutStrategyConfig:
    return BreakoutStrategyConfig(
        strategy=StrategyConfig(
            strategy_id="breakout",
            primary_timeframe=Timeframe.M5,
            warmup_bars=5,
            max_data_age=timedelta(hours=2),
        ),
        channel_lookback=5,
        breakout_buffer_atr=0.05,
        stop_buffer_atr=0.5,
        target_reward_to_risk=2.0,
        minimum_volume_ratio=0.8,
    )


def test_breakout_buy_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000, 1.1005, 1.1002, 1.1006, 1.1003, 1.1030],
        latest_open=1.1010,
        latest_low=1.1008,
        latest_high=1.1035,
        latest_volume=150,
    )
    snapshot = make_snapshot(context, atr=0.002, momentum=0.003, rsi=65)
    monkeypatch.setattr(
        "fxbot.strategy.breakout.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = BreakoutStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    ).evaluate(context)
    assert decision.action is SignalAction.BUY
    assert decision.stop_loss is not None and decision.stop_loss < decision.entry_price
    assert decision.take_profit is not None and decision.take_profit > decision.entry_price


def test_breakout_sell_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1030, 1.1025, 1.1028, 1.1024, 1.1027, 1.1000],
        latest_open=1.1020,
        latest_low=1.0995,
        latest_high=1.1022,
        latest_volume=150,
    )
    snapshot = make_snapshot(context, atr=0.002, momentum=-0.003, rsi=35)
    monkeypatch.setattr(
        "fxbot.strategy.breakout.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = BreakoutStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_DOWN),
    ).evaluate(context)
    assert decision.action is SignalAction.SELL
    assert decision.stop_loss is not None and decision.stop_loss > decision.entry_price
    assert decision.take_profit is not None and decision.take_profit < decision.entry_price


def test_breakout_rejects_bullish_liquidity_sweep(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000, 1.1005, 1.1002, 1.1006, 1.1003, 1.1004],
        latest_open=1.1002,
        latest_low=1.1000,
        latest_high=1.1020,
    )
    snapshot = make_snapshot(context, atr=0.002)
    monkeypatch.setattr(
        "fxbot.strategy.breakout.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = BreakoutStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("bullish_liquidity_sweep_rejected",)


def test_breakout_rejects_bearish_liquidity_sweep(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1030, 1.1025, 1.1028, 1.1024, 1.1027, 1.1026],
        latest_open=1.1028,
        latest_low=1.1000,
        latest_high=1.1030,
    )
    snapshot = make_snapshot(context, atr=0.002)
    monkeypatch.setattr(
        "fxbot.strategy.breakout.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = BreakoutStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_DOWN),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("bearish_liquidity_sweep_rejected",)


def test_breakout_requires_volume_confirmation(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000, 1.1005, 1.1002, 1.1006, 1.1003, 1.1030],
        latest_open=1.1010,
        latest_high=1.1035,
        latest_volume=20,
    )
    snapshot = make_snapshot(context, atr=0.002)
    monkeypatch.setattr(
        "fxbot.strategy.breakout.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = BreakoutStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("breakout_close_confirmation_missing",)


def test_breakout_filter_rejects_range_regime(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000, 1.1005, 1.1002, 1.1006, 1.1003, 1.1030],
        latest_open=1.1010,
        latest_high=1.1035,
        latest_volume=150,
    )
    snapshot = make_snapshot(context, atr=0.002)
    monkeypatch.setattr(
        "fxbot.strategy.breakout.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = BreakoutStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.RANGING),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("breakout_market_filter_rejected",)


def test_breakout_config_validation() -> None:
    base = StrategyConfig(strategy_id="breakout", primary_timeframe=Timeframe.M5)
    with pytest.raises(ValueError, match="channel_lookback"):
        BreakoutStrategyConfig(strategy=base, channel_lookback=1)
    with pytest.raises(ValueError, match="target_reward_to_risk"):
        BreakoutStrategyConfig(strategy=base, target_reward_to_risk=0.0)


def test_breakout_warmup_hold(make_context) -> None:
    context = make_context([1.10] * 5)
    decision = BreakoutStrategy(_settings()).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("breakout_channel_warmup_incomplete",)
