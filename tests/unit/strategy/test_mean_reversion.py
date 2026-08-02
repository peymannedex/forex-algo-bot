from __future__ import annotations

from datetime import timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyConfig


def _settings() -> MeanReversionConfig:
    return MeanReversionConfig(
        strategy=StrategyConfig(
            strategy_id="mean",
            primary_timeframe=Timeframe.M5,
            warmup_bars=5,
            max_data_age=timedelta(hours=2),
        ),
        band_period=5,
        stop_atr_multiple=0.05,
        minimum_reward_to_risk=0.2,
    )


def test_mean_reversion_buy_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [0.9990, 1.0010, 0.9990, 1.0010, 0.9995],
        latest_open=0.9990,
        latest_low=0.9980,
        latest_high=1.0000,
    )
    snapshot = make_snapshot(context, atr=0.001, rsi=30, fast=1.0, slow=1.0)
    monkeypatch.setattr(
        "fxbot.strategy.mean_reversion.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = MeanReversionStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.RANGING),
    ).evaluate(context)
    assert decision.action is SignalAction.BUY
    assert decision.stop_loss is not None and decision.stop_loss < decision.entry_price
    assert decision.take_profit is not None and decision.take_profit > decision.entry_price


def test_mean_reversion_sell_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [0.9990, 1.0010, 0.9990, 1.0010, 1.0005],
        latest_open=1.0010,
        latest_low=1.0000,
        latest_high=1.0020,
    )
    snapshot = make_snapshot(context, atr=0.001, rsi=70, fast=1.0, slow=1.0)
    monkeypatch.setattr(
        "fxbot.strategy.mean_reversion.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = MeanReversionStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.RANGING),
    ).evaluate(context)
    assert decision.action is SignalAction.SELL
    assert decision.take_profit is not None and decision.take_profit < decision.entry_price
    assert decision.stop_loss is not None and decision.stop_loss > decision.entry_price


def test_mean_reversion_requires_rejection_candle(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [0.9990, 1.0010, 0.9990, 1.0010, 1.0000],
        latest_open=1.0000,
        latest_low=0.9995,
        latest_high=1.0005,
    )
    snapshot = make_snapshot(context, atr=0.001, rsi=30, fast=1.0, slow=1.0)
    monkeypatch.setattr(
        "fxbot.strategy.mean_reversion.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = MeanReversionStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.RANGING),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("mean_reversion_rejection_confirmation_missing",)


def test_mean_reversion_filter_requires_range(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [0.9990, 1.0010, 0.9990, 1.0010, 0.9995],
        latest_open=0.9990,
        latest_low=0.9980,
    )
    snapshot = make_snapshot(context, atr=0.001, rsi=30)
    monkeypatch.setattr(
        "fxbot.strategy.mean_reversion.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = MeanReversionStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("mean_reversion_market_filter_rejected",)


def test_mean_reversion_rejects_zero_width_band(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context([1.0] * 5, latest_open=0.999, latest_low=0.998)
    snapshot = make_snapshot(context, atr=0.001, rsi=30)
    monkeypatch.setattr(
        "fxbot.strategy.mean_reversion.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = MeanReversionStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.RANGING),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("mean_reversion_zero_band_width",)


def test_mean_reversion_config_validation() -> None:
    base = StrategyConfig(strategy_id="mean", primary_timeframe=Timeframe.M5)
    with pytest.raises(ValueError, match="band_period"):
        MeanReversionConfig(strategy=base, band_period=1)
    with pytest.raises(ValueError, match="oversold_rsi"):
        MeanReversionConfig(strategy=base, oversold_rsi=70, overbought_rsi=60)


def test_mean_reversion_warmup_hold(make_context) -> None:
    context = make_context([1.0] * 4)
    decision = MeanReversionStrategy(_settings()).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("mean_reversion_band_warmup_incomplete",)
