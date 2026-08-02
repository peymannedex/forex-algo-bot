"""Order-block discovery, mitigation, and close-through invalidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MarketSeries
from fxbot.strategy.market_structure import StructureDirection, StructureEvent


class OrderBlockSide(StrEnum):
    """Directional order-block type."""

    BULLISH = "bullish"
    BEARISH = "bearish"


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class OrderBlockConfig:
    search_lookback: int = 10
    use_candle_body: bool = False
    minimum_break_body_atr: float = 0.80
    invalidation_buffer_atr: float = 0.05

    def __post_init__(self) -> None:
        if self.search_lookback < 1:
            raise ValueError("search_lookback must be positive")
        object.__setattr__(
            self,
            "minimum_break_body_atr",
            _non_negative(self.minimum_break_body_atr, "minimum_break_body_atr"),
        )
        object.__setattr__(
            self,
            "invalidation_buffer_atr",
            _non_negative(self.invalidation_buffer_atr, "invalidation_buffer_atr"),
        )


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """Last opposing candle before a directional structural break."""

    symbol: str
    timeframe: Timeframe
    side: OrderBlockSide
    origin_index: int
    break_index: int
    zone_low: float
    zone_high: float
    formed_at: datetime
    mitigated_at: datetime | None = None
    invalidated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", OrderBlockSide(self.side))
        if self.origin_index < 0 or self.break_index <= self.origin_index:
            raise ValueError("Order-block indices are invalid")
        low = _positive(self.zone_low, "zone_low")
        high = _positive(self.zone_high, "zone_high")
        if high <= low:
            raise ValueError("zone_high must exceed zone_low")
        object.__setattr__(self, "zone_low", low)
        object.__setattr__(self, "zone_high", high)
        formed = _utc(self.formed_at, "formed_at")
        object.__setattr__(self, "formed_at", formed)
        for name in ("mitigated_at", "invalidated_at"):
            value = getattr(self, name)
            if value is not None:
                normalized = _utc(value, name)
                if normalized < formed:
                    raise ValueError(f"{name} cannot precede formed_at")
                object.__setattr__(self, name, normalized)

    @property
    def active(self) -> bool:
        return self.invalidated_at is None

    @property
    def midpoint(self) -> float:
        return (self.zone_low + self.zone_high) / 2.0

    def overlaps(self, low: float, high: float) -> bool:
        return high >= self.zone_low and low <= self.zone_high


def detect_order_blocks(
    series: MarketSeries,
    events: tuple[StructureEvent, ...],
    *,
    atr: float,
    config: OrderBlockConfig | None = None,
) -> tuple[OrderBlock, ...]:
    """Discover order blocks from structural breaks and track their lifecycle."""

    settings = config or OrderBlockConfig()
    volatility = _positive(atr, "atr")
    invalidation_buffer = volatility * settings.invalidation_buffer_atr
    result: list[OrderBlock] = []
    used: set[tuple[OrderBlockSide, int]] = set()

    for event in events:
        break_bar = series.bars[event.index]
        break_body = abs(break_bar.mid.close - break_bar.mid.open)
        if break_body < volatility * settings.minimum_break_body_atr:
            continue
        side = (
            OrderBlockSide.BULLISH
            if event.direction is StructureDirection.BULLISH
            else OrderBlockSide.BEARISH
        )
        start = max(0, event.index - settings.search_lookback)
        origin_index: int | None = None
        for index in range(event.index - 1, start - 1, -1):
            candle = series.bars[index].mid
            opposing = (
                candle.close < candle.open
                if side is OrderBlockSide.BULLISH
                else candle.close > candle.open
            )
            if opposing:
                origin_index = index
                break
        if origin_index is None or (side, origin_index) in used:
            continue
        used.add((side, origin_index))
        origin = series.bars[origin_index].mid
        if settings.use_candle_body:
            zone_low = min(origin.open, origin.close)
            zone_high = max(origin.open, origin.close)
        else:
            zone_low = origin.low
            zone_high = origin.high

        mitigated_at: datetime | None = None
        invalidated_at: datetime | None = None
        for bar in series.bars[event.index + 1 :]:
            candle = bar.mid
            if mitigated_at is None and candle.high >= zone_low and candle.low <= zone_high:
                mitigated_at = bar.close_time
            if side is OrderBlockSide.BULLISH and candle.close < zone_low - invalidation_buffer:
                invalidated_at = bar.close_time
                break
            if side is OrderBlockSide.BEARISH and candle.close > zone_high + invalidation_buffer:
                invalidated_at = bar.close_time
                break

        result.append(
            OrderBlock(
                symbol=series.symbol,
                timeframe=series.timeframe,
                side=side,
                origin_index=origin_index,
                break_index=event.index,
                zone_low=zone_low,
                zone_high=zone_high,
                formed_at=break_bar.close_time,
                mitigated_at=mitigated_at,
                invalidated_at=invalidated_at,
            )
        )
    return tuple(result)


def nearest_order_block(
    blocks: tuple[OrderBlock, ...],
    *,
    direction: StructureDirection,
    price: float,
    active_only: bool = True,
) -> OrderBlock | None:
    """Return the latest directionally compatible block nearest current price."""

    current = _positive(price, "price")
    parsed = StructureDirection(direction)
    side = (
        OrderBlockSide.BULLISH
        if parsed is StructureDirection.BULLISH
        else OrderBlockSide.BEARISH
        if parsed is StructureDirection.BEARISH
        else None
    )
    if side is None:
        return None
    candidates = [
        block
        for block in blocks
        if block.side is side and (block.active or not active_only)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda block: abs(block.midpoint - current))
