"""Strategy protocol, execution wrapper, and duplicate-signal suppression."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.models import SignalAction, StrategyConfig, StrategyDecision


class StrategyContractError(RuntimeError):
    """Raised when a strategy violates its declared output contract."""


class Strategy(Protocol):
    """Structural interface implemented by manual and algorithmic strategies."""

    @property
    def config(self) -> StrategyConfig: ...

    def evaluate(self, context: MultiTimeframeContext) -> StrategyDecision: ...


class SignalDeduplicator:
    """Suppress semantically identical signals inside a configured time window."""

    def __init__(self) -> None:
        self._last: dict[tuple[str, str, str, str], tuple[datetime, str]] = {}

    def accept(self, decision: StrategyDecision, window_seconds: float) -> bool:
        if window_seconds < 0.0:
            raise ValueError("window_seconds cannot be negative")
        if decision.action is SignalAction.HOLD or window_seconds == 0.0:
            return True
        key = (
            decision.strategy_id,
            decision.symbol,
            decision.timeframe.value,
            decision.action.value,
        )
        previous = self._last.get(key)
        if previous is not None:
            previous_time, previous_fingerprint = previous
            elapsed = (decision.as_of - previous_time).total_seconds()
            if elapsed < 0.0:
                return False
            if elapsed <= window_seconds and previous_fingerprint == decision.semantic_fingerprint:
                return False
        self._last[key] = (decision.as_of, decision.semantic_fingerprint)
        return True

    def clear(self) -> None:
        self._last.clear()


class StrategyRuntime:
    """Apply common readiness, confidence, and duplicate safeguards."""

    def __init__(self, deduplicator: SignalDeduplicator | None = None) -> None:
        self.deduplicator = deduplicator or SignalDeduplicator()

    def run(
        self,
        strategy: Strategy,
        context: MultiTimeframeContext,
    ) -> StrategyDecision:
        config = strategy.config
        issues = context.validate(config)
        if issues:
            return StrategyDecision.hold(
                strategy_id=config.strategy_id,
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                as_of=context.as_of,
                reason="; ".join(issue.code.value for issue in issues),
                metadata=tuple(
                    (f"issue_{index}", issue.message) for index, issue in enumerate(issues)
                ),
            )

        decision = strategy.evaluate(context)
        self._validate_contract(config, context, decision)
        if (
            decision.action is not SignalAction.HOLD
            and decision.confidence < config.min_confidence
        ):
            return StrategyDecision.hold(
                strategy_id=config.strategy_id,
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                as_of=context.as_of,
                reason="signal_below_minimum_confidence",
                regime=decision.regime,
                metadata=(("original_action", decision.action.value),),
            )

        accepted = self.deduplicator.accept(
            decision,
            config.duplicate_suppression_window.total_seconds(),
        )
        if not accepted:
            return StrategyDecision.hold(
                strategy_id=config.strategy_id,
                symbol=context.symbol,
                timeframe=context.primary_timeframe,
                as_of=context.as_of,
                reason="duplicate_signal_suppressed",
                regime=decision.regime,
                metadata=(("original_action", decision.action.value),),
            )
        return decision

    @staticmethod
    def _validate_contract(
        config: StrategyConfig,
        context: MultiTimeframeContext,
        decision: StrategyDecision,
    ) -> None:
        mismatches: list[str] = []
        if decision.strategy_id != config.strategy_id:
            mismatches.append("strategy_id")
        if decision.symbol != context.symbol:
            mismatches.append("symbol")
        if decision.timeframe is not context.primary_timeframe:
            mismatches.append("timeframe")
        if decision.as_of != context.as_of:
            mismatches.append("as_of")
        if mismatches:
            raise StrategyContractError(
                "Strategy decision does not match runtime context: " + ", ".join(mismatches)
            )
