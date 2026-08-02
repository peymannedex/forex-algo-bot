"""Reusable market-quality, session, volatility, and regime filters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import time
from math import isfinite

from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.models import (
    IndicatorSnapshot,
    MarketRegime,
    RegimeAssessment,
    RegimeConfluence,
    SignalAction,
)


def _fraction(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _weekday_default() -> frozenset[int]:
    return frozenset(range(5))


@dataclass(frozen=True, slots=True)
class StrategyFilterConfig:
    """Cross-strategy execution filters applied before signal generation.

    Session times are interpreted as UTC wall-clock times.  When ``session_end``
    is earlier than ``session_start``, the interval is treated as an overnight
    session spanning midnight.
    """

    allowed_regimes: tuple[MarketRegime, ...] = ()
    blocked_regimes: tuple[MarketRegime, ...] = (MarketRegime.ILLIQUID,)
    max_spread_to_atr: float = 0.30
    min_atr_fraction: float = 0.0
    max_atr_fraction: float = 1.0
    minimum_alignment_score: float = 0.0
    require_directional_alignment: bool = False
    allowed_weekdays: frozenset[int] = field(default_factory=_weekday_default)
    session_start: time | None = None
    session_end: time | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_regimes",
            tuple(MarketRegime(item) for item in self.allowed_regimes),
        )
        object.__setattr__(
            self,
            "blocked_regimes",
            tuple(MarketRegime(item) for item in self.blocked_regimes),
        )
        object.__setattr__(
            self,
            "max_spread_to_atr",
            _non_negative(self.max_spread_to_atr, "max_spread_to_atr"),
        )
        object.__setattr__(
            self,
            "min_atr_fraction",
            _non_negative(self.min_atr_fraction, "min_atr_fraction"),
        )
        object.__setattr__(
            self,
            "max_atr_fraction",
            _positive(self.max_atr_fraction, "max_atr_fraction"),
        )
        if self.max_atr_fraction < self.min_atr_fraction:
            raise ValueError("max_atr_fraction cannot be below min_atr_fraction")
        object.__setattr__(
            self,
            "minimum_alignment_score",
            _fraction(self.minimum_alignment_score, "minimum_alignment_score"),
        )
        weekdays = frozenset(int(day) for day in self.allowed_weekdays)
        if not weekdays:
            raise ValueError("allowed_weekdays cannot be empty")
        if any(day < 0 or day > 6 for day in weekdays):
            raise ValueError("allowed_weekdays values must be between 0 and 6")
        object.__setattr__(self, "allowed_weekdays", weekdays)
        if (self.session_start is None) != (self.session_end is None):
            raise ValueError("session_start and session_end must be configured together")
        for name in ("session_start", "session_end"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is not None:
                raise ValueError(f"{name} must be a naive UTC wall-clock time")


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Auditable result of the common market-quality gate."""

    allowed: bool
    reasons: tuple[str, ...]
    confidence_multiplier: float
    metrics: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_multiplier",
            _fraction(self.confidence_multiplier, "confidence_multiplier"),
        )
        if self.allowed and self.reasons:
            raise ValueError("Allowed filter results cannot contain rejection reasons")
        if not self.allowed and not self.reasons:
            raise ValueError("Rejected filter results require at least one reason")
        for name, value in self.metrics:
            if not name.strip():
                raise ValueError("Filter metric names cannot be empty")
            if not isfinite(float(value)):
                raise ValueError(f"Filter metric {name} must be finite")


class MarketFilter:
    """Evaluate spread, volatility, session, and regime eligibility."""

    def __init__(self, config: StrategyFilterConfig | None = None) -> None:
        self.config = config or StrategyFilterConfig()

    def evaluate(
        self,
        *,
        context: MultiTimeframeContext,
        indicators: IndicatorSnapshot,
        regime: RegimeAssessment,
        confluence: RegimeConfluence | None = None,
        expected_action: SignalAction | None = None,
    ) -> FilterResult:
        reasons: list[str] = []
        spread_ratio = indicators.value("spread_to_atr")
        atr_fraction = indicators.value("atr_fraction")

        if context.as_of.weekday() not in self.config.allowed_weekdays:
            reasons.append("weekday_not_allowed")
        if not self._inside_session(context):
            reasons.append("outside_session")
        if regime.regime in self.config.blocked_regimes:
            reasons.append(f"blocked_regime:{regime.regime.value}")
        if self.config.allowed_regimes and regime.regime not in self.config.allowed_regimes:
            reasons.append(f"regime_not_allowed:{regime.regime.value}")
        if spread_ratio > self.config.max_spread_to_atr:
            reasons.append("spread_to_atr_exceeds_limit")
        if atr_fraction < self.config.min_atr_fraction:
            reasons.append("atr_fraction_below_minimum")
        if atr_fraction > self.config.max_atr_fraction:
            reasons.append("atr_fraction_exceeds_maximum")

        if self.config.require_directional_alignment:
            if confluence is None:
                reasons.append("regime_confluence_required")
            else:
                if confluence.alignment_score < self.config.minimum_alignment_score:
                    reasons.append("regime_alignment_below_minimum")
                expected_regime = _expected_regime(expected_action)
                if expected_regime is not None and confluence.dominant_regime is not expected_regime:
                    reasons.append("regime_direction_mismatch")

        metrics = (
            ("atr_fraction", atr_fraction),
            (
                "regime_alignment",
                confluence.alignment_score if confluence is not None else 0.0,
            ),
            ("spread_to_atr", spread_ratio),
        )
        if reasons:
            return FilterResult(
                allowed=False,
                reasons=tuple(reasons),
                confidence_multiplier=0.0,
                metrics=metrics,
            )

        spread_quality = 1.0
        if self.config.max_spread_to_atr > 0.0:
            spread_quality = 1.0 - min(
                1.0,
                spread_ratio / self.config.max_spread_to_atr,
            ) * 0.25
        alignment_quality = (
            confluence.alignment_score
            if confluence is not None and self.config.require_directional_alignment
            else 1.0
        )
        return FilterResult(
            allowed=True,
            reasons=(),
            confidence_multiplier=max(0.0, min(1.0, spread_quality * alignment_quality)),
            metrics=metrics,
        )

    def _inside_session(self, context: MultiTimeframeContext) -> bool:
        start = self.config.session_start
        end = self.config.session_end
        if start is None or end is None:
            return True
        current = context.as_of.time().replace(tzinfo=None)
        if start == end:
            return True
        if start < end:
            return start <= current < end
        return current >= start or current < end


def clamp_confidence(value: float) -> float:
    """Clamp a finite confidence score to the canonical ``[0, 1]`` range."""

    number = float(value)
    if not isfinite(number):
        raise ValueError("confidence must be finite")
    return max(0.0, min(1.0, number))


def atr_bracket(
    *,
    action: SignalAction,
    entry_price: float,
    atr: float,
    stop_multiple: float,
    target_multiple: float,
) -> tuple[float, float]:
    """Return directionally valid ATR stop and target prices."""

    entry = _positive(entry_price, "entry_price")
    volatility = _positive(atr, "atr")
    stop_distance = volatility * _positive(stop_multiple, "stop_multiple")
    target_distance = volatility * _positive(target_multiple, "target_multiple")
    if action is SignalAction.BUY:
        stop = entry - stop_distance
        target = entry + target_distance
    elif action is SignalAction.SELL:
        stop = entry + stop_distance
        target = entry - target_distance
    else:
        raise ValueError("ATR brackets require BUY or SELL action")
    if stop <= 0.0 or target <= 0.0:
        raise ValueError("ATR bracket produced a non-positive executable price")
    return stop, target


def format_metadata(values: Mapping[str, float | int | str]) -> tuple[tuple[str, str], ...]:
    """Return stable, sorted strategy metadata suitable for audit logs."""

    normalized: list[tuple[str, str]] = []
    for key, value in values.items():
        name = key.strip()
        if not name:
            raise ValueError("Metadata keys cannot be empty")
        if isinstance(value, float):
            if not isfinite(value):
                raise ValueError(f"Metadata value {name} must be finite")
            rendered = f"{value:.12g}"
        else:
            rendered = str(value)
        normalized.append((name, rendered))
    return tuple(sorted(normalized))


def _expected_regime(action: SignalAction | None) -> MarketRegime | None:
    if action is SignalAction.BUY:
        return MarketRegime.TRENDING_UP
    if action is SignalAction.SELL:
        return MarketRegime.TRENDING_DOWN
    return None
