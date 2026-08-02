"""Immutable strategy-domain models and configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import isfinite

from fxbot.domain.enums import Timeframe


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _fraction(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return number


def _positive_optional(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


class SignalAction(StrEnum):
    """Canonical strategy actions consumed by backtest and execution layers."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"


class MarketRegime(StrEnum):
    """Coarse market-state classification used for strategy filtering."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Common safety and readiness settings shared by every strategy."""

    strategy_id: str
    primary_timeframe: Timeframe
    required_timeframes: tuple[Timeframe, ...] = ()
    warmup_bars: int = 50
    max_data_age: timedelta = timedelta(minutes=10)
    min_confidence: float = 0.0
    duplicate_suppression_window: timedelta = timedelta(0)
    allow_incomplete_bars: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", _identifier(self.strategy_id, "strategy_id"))
        primary = Timeframe.parse(self.primary_timeframe)
        if primary is Timeframe.TICK:
            raise ValueError("primary_timeframe must be a bar timeframe")
        object.__setattr__(self, "primary_timeframe", primary)

        ordered: list[Timeframe] = [primary]
        for raw in self.required_timeframes:
            timeframe = Timeframe.parse(raw)
            if timeframe is Timeframe.TICK:
                raise ValueError("required_timeframes cannot contain tick")
            if timeframe not in ordered:
                ordered.append(timeframe)
        object.__setattr__(self, "required_timeframes", tuple(ordered))

        if self.warmup_bars < 1:
            raise ValueError("warmup_bars must be at least 1")
        if self.max_data_age <= timedelta(0):
            raise ValueError("max_data_age must be positive")
        object.__setattr__(
            self,
            "min_confidence",
            _fraction(self.min_confidence, "min_confidence"),
        )
        if self.duplicate_suppression_window < timedelta(0):
            raise ValueError("duplicate_suppression_window cannot be negative")


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Deterministic and auditable output of one strategy evaluation."""

    strategy_id: str
    symbol: str
    timeframe: Timeframe
    action: SignalAction
    as_of: datetime
    confidence: float
    reasons: tuple[str, ...]
    regime: MarketRegime = MarketRegime.UNKNOWN
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", _identifier(self.strategy_id, "strategy_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        timeframe = Timeframe.parse(self.timeframe)
        if timeframe is Timeframe.TICK:
            raise ValueError("Strategy decisions require a bar timeframe")
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "action", SignalAction(self.action))
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        object.__setattr__(self, "confidence", _fraction(self.confidence, "confidence"))
        object.__setattr__(self, "regime", MarketRegime(self.regime))
        object.__setattr__(
            self,
            "entry_price",
            _positive_optional(self.entry_price, "entry_price"),
        )
        object.__setattr__(
            self,
            "stop_loss",
            _positive_optional(self.stop_loss, "stop_loss"),
        )
        object.__setattr__(
            self,
            "take_profit",
            _positive_optional(self.take_profit, "take_profit"),
        )

        reasons = tuple(reason.strip() for reason in self.reasons if reason.strip())
        if not reasons:
            raise ValueError("reasons must contain at least one non-empty reason")
        object.__setattr__(self, "reasons", reasons)

        metadata: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_key, raw_value in self.metadata:
            key = _identifier(raw_key, "metadata key")
            if key in seen:
                raise ValueError(f"Duplicate metadata key: {key}")
            seen.add(key)
            metadata.append((key, str(raw_value)))
        object.__setattr__(self, "metadata", tuple(sorted(metadata)))

        if self.action is SignalAction.HOLD and any(
            value is not None for value in (self.entry_price, self.stop_loss, self.take_profit)
        ):
            raise ValueError("HOLD decisions cannot contain executable prices")
        if self.action is SignalAction.EXIT and any(
            value is not None for value in (self.stop_loss, self.take_profit)
        ):
            raise ValueError("EXIT decisions cannot contain stop_loss or take_profit")

    @property
    def semantic_fingerprint(self) -> str:
        """Hash decision meaning while intentionally excluding evaluation time."""

        payload = repr(
            (
                self.strategy_id,
                self.symbol,
                self.timeframe.value,
                self.action.value,
                round(self.confidence, 12),
                self.reasons,
                self.regime.value,
                self.entry_price,
                self.stop_loss,
                self.take_profit,
                self.metadata,
            )
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    @classmethod
    def hold(
        cls,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: Timeframe,
        as_of: datetime,
        reason: str,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> StrategyDecision:
        """Build a standardized non-executable decision."""

        return cls(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            action=SignalAction.HOLD,
            as_of=as_of,
            confidence=0.0,
            reasons=(reason,),
            regime=regime,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Named indicator values calculated from data available at ``as_of``."""

    symbol: str
    timeframe: Timeframe
    as_of: datetime
    values: tuple[tuple[str, float], ...]
    sample_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        timeframe = Timeframe.parse(self.timeframe)
        if timeframe is Timeframe.TICK:
            raise ValueError("IndicatorSnapshot timeframe cannot be tick")
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        if self.sample_size < 1:
            raise ValueError("sample_size must be positive")

        normalized: list[tuple[str, float]] = []
        seen: set[str] = set()
        for raw_name, raw_value in self.values:
            name = _identifier(raw_name, "indicator name")
            value = float(raw_value)
            if name in seen:
                raise ValueError(f"Duplicate indicator name: {name}")
            if not isfinite(value):
                raise ValueError(f"Indicator {name} must be finite")
            seen.add(name)
            normalized.append((name, value))
        object.__setattr__(self, "values", tuple(sorted(normalized)))

    def value(self, name: str) -> float:
        """Return one named indicator value or raise a precise key error."""

        try:
            return dict(self.values)[name]
        except KeyError as exc:
            raise KeyError(f"Indicator not available: {name}") from exc


@dataclass(frozen=True, slots=True)
class RegimeAssessment:
    """Regime classification with normalized evidence and audit reasons."""

    symbol: str
    timeframe: Timeframe
    as_of: datetime
    regime: MarketRegime
    confidence: float
    metrics: tuple[tuple[str, float], ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        timeframe = Timeframe.parse(self.timeframe)
        if timeframe is Timeframe.TICK:
            raise ValueError("RegimeAssessment timeframe cannot be tick")
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        object.__setattr__(self, "regime", MarketRegime(self.regime))
        object.__setattr__(self, "confidence", _fraction(self.confidence, "confidence"))
        if not self.reasons:
            raise ValueError("reasons cannot be empty")
        for name, value in self.metrics:
            _identifier(name, "metric name")
            if not isfinite(float(value)):
                raise ValueError(f"Metric {name} must be finite")

    def metric(self, name: str) -> float:
        try:
            return dict(self.metrics)[name]
        except KeyError as exc:
            raise KeyError(f"Regime metric not available: {name}") from exc


@dataclass(frozen=True, slots=True)
class RegimeConfluence:
    """Multi-timeframe agreement summary for directional regime filters."""

    symbol: str
    as_of: datetime
    primary_timeframe: Timeframe
    primary_regime: MarketRegime
    dominant_regime: MarketRegime
    alignment_score: float
    assessments: tuple[tuple[Timeframe, MarketRegime], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        primary = Timeframe.parse(self.primary_timeframe)
        if primary is Timeframe.TICK:
            raise ValueError("primary_timeframe cannot be tick")
        object.__setattr__(self, "primary_timeframe", primary)
        object.__setattr__(self, "primary_regime", MarketRegime(self.primary_regime))
        object.__setattr__(self, "dominant_regime", MarketRegime(self.dominant_regime))
        object.__setattr__(
            self,
            "alignment_score",
            _fraction(self.alignment_score, "alignment_score"),
        )
        if not self.assessments:
            raise ValueError("assessments cannot be empty")

    @property
    def directionally_aligned(self) -> bool:
        return self.dominant_regime in {
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
        } and self.alignment_score >= 0.5
