from __future__ import annotations

from datetime import timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyConfig
from fxbot.strategy.momentum import MomentumStrategy, MomentumStrategyConfig


def _settings() -> MomentumStrategyConfig:
    return MomentumStrategyConfig(
        strategy=StrategyConfig(
            strategy_id="momentum",
            primary_timeframe=Timeframe.M5,
            required_timeframes=(Timeframe.H1,),
            warmup_bars=20,
            max_data_age=timedelta(hours=2),
        )
    )


def test_momentum_buy_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1030],
        higher_closes=[1.1000] * 30,
        latest_open=1.1010,
        latest_low=1.1008,
        latest_high=1.1034,
    )
    snapshot = make_snapshot(
        context,
        atr=0.002,
        fast=1.102,
        slow=1.100,
        momentum=0.003,
        rsi=65,
    )
    monkeypatch.setattr(
        "fxbot.strategy.momentum.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    strategy = MomentumStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    )
    decision = strategy.evaluate(context)
    assert decision.action is SignalAction.BUY
    assert decision.confidence > 0.0
    assert decision.stop_loss is not None and decision.stop_loss < decision.entry_price


def test_momentum_sell_signal(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1030] * 29 + [1.1000],
        higher_closes=[1.1030] * 30,
        latest_open=1.1020,
        latest_low=1.0995,
        latest_high=1.1022,
    )
    snapshot = make_snapshot(
        context,
        atr=0.002,
        fast=1.1005,
        slow=1.1020,
        momentum=-0.003,
        rsi=35,
    )
    monkeypatch.setattr(
        "fxbot.strategy.momentum.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    strategy = MomentumStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_DOWN),
    )
    decision = strategy.evaluate(context)
    assert decision.action is SignalAction.SELL
    assert decision.take_profit is not None and decision.take_profit < decision.entry_price


def test_momentum_holds_when_thresholds_are_not_met(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1001],
        higher_closes=[1.1000] * 30,
        latest_open=1.1000,
    )
    snapshot = make_snapshot(context, momentum=0.0001, rsi=50, fast=1.1001, slow=1.1000)
    monkeypatch.setattr(
        "fxbot.strategy.momentum.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    decision = MomentumStrategy(
        _settings(),
        regime_classifier=classifier_factory(MarketRegime.TRENDING_UP),
    ).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("momentum_thresholds_not_met",)


def test_momentum_holds_on_higher_timeframe_mismatch(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1030],
        higher_closes=[1.1000] * 30,
        latest_open=1.1010,
    )
    snapshot = make_snapshot(context, momentum=0.003, rsi=65, fast=1.102, slow=1.100)
    monkeypatch.setattr(
        "fxbot.strategy.momentum.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    classifier = classifier_factory(
        MarketRegime.TRENDING_UP,
        dominant=MarketRegime.TRENDING_DOWN,
    )
    decision = MomentumStrategy(_settings(), regime_classifier=classifier).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("momentum_higher_timeframe_not_confirmed",)


def test_momentum_filter_blocks_ranging_market(
    monkeypatch,
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context(
        [1.1000] * 29 + [1.1030],
        higher_closes=[1.1000] * 30,
        latest_open=1.1010,
    )
    snapshot = make_snapshot(context, momentum=0.003, rsi=65, fast=1.102, slow=1.100)
    monkeypatch.setattr(
        "fxbot.strategy.momentum.calculate_indicators",
        lambda *_args, **_kwargs: snapshot,
    )
    classifier = classifier_factory(
        MarketRegime.RANGING,
        dominant=MarketRegime.TRENDING_UP,
    )
    decision = MomentumStrategy(_settings(), regime_classifier=classifier).evaluate(context)
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("momentum_market_filter_rejected",)


def test_momentum_config_requires_higher_timeframe() -> None:
    with pytest.raises(ValueError, match="Higher-timeframe"):
        MomentumStrategyConfig(
            strategy=StrategyConfig(
                strategy_id="momentum",
                primary_timeframe=Timeframe.M5,
            )
        )


def test_momentum_can_disable_higher_timeframe_requirement() -> None:
    config = MomentumStrategyConfig(
        strategy=StrategyConfig(
            strategy_id="momentum",
            primary_timeframe=Timeframe.M5,
        ),
        require_higher_timeframe_confirmation=False,
    )
    assert not config.require_higher_timeframe_confirmation
