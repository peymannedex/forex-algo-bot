"""Three-candle fair-value gaps with deterministic fill tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MarketSeries
from fxbot.strategy.market_structure import StructureDirection


class ImbalanceSide(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ImbalanceConfig:
    minimum_gap_atr: float = 0.05

    def __post_init__(self) -> None:
        value = float(self.minimum_gap_atr)
        if not isfinite(value) or value < 0.0:
            raise ValueError("minimum_gap_atr must be finite and non-negative")
        object.__setattr__(self, "minimum_gap_atr", value)


@dataclass(frozen=True, slots=True)
class FairValueGap:
    """Untraded three-candle price interval and its later fill state."""

    symbol: str
    timeframe: Timeframe
    side: ImbalanceSide
    created_index: int
    gap_low: float
    gap_high: float
    formed_at: datetime
    fill_fraction: float = 0.0
    filled_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", ImbalanceSide(self.side))
        if self.created_index < 2:
            raise ValueError("created_index must be at least 2")
        low = _positive(self.gap_low, "gap_low")
        high = _positive(self.gap_high, "gap_high")
        if high <= low:
            raise ValueError("gap_high must exceed gap_low")
        object.__setattr__(self, "gap_low", low)
        object.__setattr__(self, "gap_high", high)
        object.__setattr__(self, "formed_at", _utc(self.formed_at, "formed_at"))
        fraction = float(self.fill_fraction)
        if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("fill_fraction must be between 0 and 1")
        object.__setattr__(self, "fill_fraction", fraction)
        if self.filled_at is not None:
            object.__setattr__(self, "filled_at", _utc(self.filled_at, "filled_at"))
            if fraction < 1.0:
                raise ValueError("filled_at requires fill_fraction=1")

    @property
    def active(self) -> bool:
        return self.filled_at is None

    @property
    def midpoint(self) -> float:
        return (self.gap_low + self.gap_high) / 2.0

    def overlaps(self, low: float, high: float) -> bool:
        return high >= self.gap_low and low <= self.gap_high


def detect_fair_value_gaps(
    series: MarketSeries,
    *,
    atr: float,
    config: ImbalanceConfig | None = None,
) -> tuple[FairValueGap, ...]:
    """Detect look-ahead-safe FVGs and calculate later partial/full fills."""

    settings = config or ImbalanceConfig()
    volatility = _positive(atr, "atr")
    minimum = volatility * settings.minimum_gap_atr
    result: list[FairValueGap] = []

    for index in range(2, len(series.bars)):
        first = series.bars[index - 2].mid
        third = series.bars[index].mid
        side: ImbalanceSide | None = None
        gap_low = 0.0
        gap_high = 0.0
        if third.low > first.high and third.low - first.high >= minimum:
            side = ImbalanceSide.BULLISH
            gap_low = first.high
            gap_high = third.low
        elif third.high < first.low and first.low - third.high >= minimum:
            side = ImbalanceSide.BEARISH
            gap_low = third.high
            gap_high = first.low
        if side is None:
            continue

        size = gap_high - gap_low
        fill_fraction = 0.0
        filled_at: datetime | None = None
        for bar in series.bars[index + 1 :]:
            candle = bar.mid
            if side is ImbalanceSide.BULLISH and candle.low < gap_high:
                penetrated = gap_high - max(candle.low, gap_low)
                fill_fraction = max(fill_fraction, min(1.0, penetrated / size))
                if candle.low <= gap_low:
                    fill_fraction = 1.0
                    filled_at = bar.close_time
                    break
            elif side is ImbalanceSide.BEARISH and candle.high > gap_low:
                penetrated = min(candle.high, gap_high) - gap_low
                fill_fraction = max(fill_fraction, min(1.0, penetrated / size))
                if candle.high >= gap_high:
                    fill_fraction = 1.0
                    filled_at = bar.close_time
                    break

        result.append(
            FairValueGap(
                symbol=series.symbol,
                timeframe=series.timeframe,
                side=side,
                created_index=index,
                gap_low=gap_low,
                gap_high=gap_high,
                formed_at=series.bars[index].close_time,
                fill_fraction=fill_fraction,
                filled_at=filled_at,
            )
        )
    return tuple(result)


def nearest_fair_value_gap(
    gaps: tuple[FairValueGap, ...],
    *,
    direction: StructureDirection,
    price: float,
    active_only: bool = True,
) -> FairValueGap | None:
    """Return the nearest directionally compatible FVG."""

    current = _positive(price, "price")
    parsed = StructureDirection(direction)
    side = (
        ImbalanceSide.BULLISH
        if parsed is StructureDirection.BULLISH
        else ImbalanceSide.BEARISH
        if parsed is StructureDirection.BEARISH
        else None
    )
    if side is None:
        return None
    candidates = [gap for gap in gaps if gap.side is side and (gap.active or not active_only)]
    if not candidates:
        return None
    return min(candidates, key=lambda gap: abs(gap.midpoint - current))
