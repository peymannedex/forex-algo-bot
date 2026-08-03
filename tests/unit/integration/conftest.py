from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.execution.models import Quote
from fxbot.integration.models import PaperFrame
from fxbot.strategy.context import MarketSeries, MultiTimeframeContext
from fxbot.strategy.models import (
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)

BASE = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


def make_bar(
    index: int = 0,
    *,
    bid_open: float = 1.1000,
    bid_close: float = 1.1001,
) -> Bar:
    spread = 0.0002
    high = max(bid_open, bid_close) + 0.0002
    low = min(bid_open, bid_close) - 0.0002
    return Bar(
        symbol="EURUSD",
        open_time=BASE + timedelta(minutes=5 * index),
        timeframe=Timeframe.M5,
        bid=OHLC(bid_open, high, low, bid_close),
        ask=OHLC(
            bid_open + spread,
            high + spread,
            low + spread,
            bid_close + spread,
        ),
        tick_volume=100,
    )


def make_frame(index: int = 0) -> PaperFrame:
    bars = tuple(make_bar(item) for item in range(index + 1))
    as_of = bars[-1].close_time
    context = MultiTimeframeContext(
        symbol="EURUSD",
        as_of=as_of,
        primary_timeframe=Timeframe.M5,
        series=(MarketSeries("EURUSD", Timeframe.M5, bars),),
    )
    return PaperFrame(
        context=context,
        quote=Quote(
            "EURUSD",
            bars[-1].bid.close,
            bars[-1].ask.close,
            as_of,
        ),
    )


@dataclass
class StaticStrategy:
    action: SignalAction
    confidence: float = 0.9
    stop_loss: float | None = None
    take_profit: float | None = None

    @property
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_id="static",
            primary_timeframe=Timeframe.M5,
            warmup_bars=1,
        )

    def evaluate(self, context: MultiTimeframeContext) -> StrategyDecision:
        if self.action is SignalAction.HOLD:
            return StrategyDecision.hold(
                strategy_id="static",
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                as_of=context.as_of,
                reason="hold",
            )
        return StrategyDecision(
            strategy_id="static",
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            action=self.action,
            as_of=context.as_of,
            confidence=self.confidence,
            reasons=("test",),
            entry_price=context.primary.latest.mid.close,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
        )


@pytest.fixture
def frame() -> PaperFrame:
    return make_frame()
