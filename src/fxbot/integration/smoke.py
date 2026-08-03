"""Deterministic non-production strategy used only to exercise acceptance paths."""

from __future__ import annotations

from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.models import (
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)


class AcceptanceSmokeStrategy:
    """Open, close, reverse, and close on fixed primary-bar counts."""

    def __init__(self, config: StrategyConfig) -> None:
        self._config = config

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def evaluate(self, context: MultiTimeframeContext) -> StrategyDecision:
        count = len(context.primary.bars)
        phase = count % 40
        close = context.primary.latest.mid.close
        distance = max(close * 0.001, 0.0005)
        if phase == 5:
            return StrategyDecision(
                strategy_id=self.config.strategy_id,
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                action=SignalAction.BUY,
                as_of=context.as_of,
                confidence=1.0,
                reasons=("acceptance_smoke_buy",),
                entry_price=close,
                stop_loss=close - distance,
                take_profit=close + distance * 2.0,
            )
        if phase == 25:
            return StrategyDecision(
                strategy_id=self.config.strategy_id,
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                action=SignalAction.SELL,
                as_of=context.as_of,
                confidence=1.0,
                reasons=("acceptance_smoke_sell",),
                entry_price=close,
                stop_loss=close + distance,
                take_profit=close - distance * 2.0,
            )
        if phase in {15, 35}:
            return StrategyDecision(
                strategy_id=self.config.strategy_id,
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                action=SignalAction.EXIT,
                as_of=context.as_of,
                confidence=1.0,
                reasons=("acceptance_smoke_exit",),
                entry_price=close,
            )
        return StrategyDecision.hold(
            strategy_id=self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            as_of=context.as_of,
            reason="acceptance_smoke_hold",
        )
