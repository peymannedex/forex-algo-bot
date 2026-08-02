from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.strategy.base import (
    SignalDeduplicator,
    StrategyContractError,
    StrategyRuntime,
)
from fxbot.strategy.context import MarketSeries, MultiTimeframeContext
from fxbot.strategy.models import SignalAction, StrategyConfig, StrategyDecision

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_context(count: int = 5) -> MultiTimeframeContext:
    bars = tuple(
        Bar(
            symbol="EURUSD",
            open_time=BASE + timedelta(minutes=5 * index),
            timeframe=Timeframe.M5,
            bid=OHLC(1.0999, 1.1001, 1.0997, 1.0999),
            ask=OHLC(1.1001, 1.1003, 1.0999, 1.1001),
        )
        for index in range(count)
    )
    return MultiTimeframeContext(
        symbol="EURUSD",
        as_of=bars[-1].close_time,
        primary_timeframe=Timeframe.M5,
        series=(MarketSeries("EURUSD", Timeframe.M5, bars),),
    )


@dataclass
class StubStrategy:
    config: StrategyConfig
    action: SignalAction = SignalAction.BUY
    confidence: float = 0.8
    strategy_id_override: str | None = None

    def evaluate(self, context: MultiTimeframeContext) -> StrategyDecision:
        return StrategyDecision(
            strategy_id=self.strategy_id_override or self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            action=self.action,
            as_of=context.as_of,
            confidence=self.confidence,
            reasons=("stub",),
        )


def test_runtime_returns_hold_when_context_not_ready() -> None:
    strategy = StubStrategy(
        StrategyConfig(
            strategy_id="s",
            primary_timeframe=Timeframe.M5,
            warmup_bars=10,
            max_data_age=timedelta(minutes=10),
        )
    )
    decision = StrategyRuntime().run(strategy, make_context(3))
    assert decision.action is SignalAction.HOLD
    assert "insufficient_warmup" in decision.reasons[0]


def test_runtime_converts_low_confidence_signal_to_hold() -> None:
    strategy = StubStrategy(
        StrategyConfig(
            strategy_id="s",
            primary_timeframe=Timeframe.M5,
            warmup_bars=1,
            min_confidence=0.75,
            max_data_age=timedelta(minutes=10),
        ),
        confidence=0.5,
    )
    decision = StrategyRuntime().run(strategy, make_context())
    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("signal_below_minimum_confidence",)


def test_runtime_suppresses_duplicate_signal() -> None:
    config = StrategyConfig(
        strategy_id="s",
        primary_timeframe=Timeframe.M5,
        warmup_bars=1,
        max_data_age=timedelta(minutes=10),
        duplicate_suppression_window=timedelta(hours=1),
    )
    strategy = StubStrategy(config)
    runtime = StrategyRuntime()
    context = make_context()
    assert runtime.run(strategy, context).action is SignalAction.BUY
    assert runtime.run(strategy, context).action is SignalAction.HOLD


def test_runtime_rejects_contract_mismatch() -> None:
    strategy = StubStrategy(
        StrategyConfig(
            strategy_id="s",
            primary_timeframe=Timeframe.M5,
            warmup_bars=1,
            max_data_age=timedelta(minutes=10),
        ),
        strategy_id_override="wrong",
    )
    with pytest.raises(StrategyContractError, match="strategy_id"):
        StrategyRuntime().run(strategy, make_context())


def test_deduplicator_allows_hold_and_clear() -> None:
    deduplicator = SignalDeduplicator()
    hold = StrategyDecision.hold(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        as_of=BASE,
        reason="hold",
    )
    assert deduplicator.accept(hold, 60.0)
    assert deduplicator.accept(hold, 60.0)
    deduplicator.clear()


def test_deduplicator_rejects_out_of_order_signal() -> None:
    deduplicator = SignalDeduplicator()
    newer = StrategyDecision(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        action=SignalAction.BUY,
        as_of=BASE + timedelta(minutes=5),
        confidence=0.8,
        reasons=("same",),
    )
    older = StrategyDecision(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        action=SignalAction.BUY,
        as_of=BASE,
        confidence=0.8,
        reasons=("same",),
    )
    assert deduplicator.accept(newer, 60.0)
    assert not deduplicator.accept(older, 60.0)
