"""Look-ahead-safe swing, structure-break, displacement, and dealing-range analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MarketSeries, MultiTimeframeContext


class SwingKind(StrEnum):
    """Confirmed local-extremum type."""

    HIGH = "high"
    LOW = "low"


class StructureDirection(StrEnum):
    """Directional market-structure state."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureEventKind(StrEnum):
    """Canonical structural break classification."""

    BREAK_OF_STRUCTURE = "bos"
    CHANGE_OF_CHARACTER = "choch"


class DealingRangeZone(StrEnum):
    """Location of price inside the latest confirmed dealing range."""

    PREMIUM = "premium"
    EQUILIBRIUM = "equilibrium"
    DISCOUNT = "discount"
    UNKNOWN = "unknown"


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
class SwingPoint:
    """A pivot that becomes usable only after ``right_bars`` have closed."""

    symbol: str
    timeframe: Timeframe
    index: int
    kind: SwingKind
    price: float
    bar_time: datetime
    confirmed_at: datetime

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        object.__setattr__(self, "symbol", symbol)
        timeframe = Timeframe.parse(self.timeframe)
        if timeframe is Timeframe.TICK:
            raise ValueError("SwingPoint timeframe cannot be tick")
        object.__setattr__(self, "timeframe", timeframe)
        if self.index < 0:
            raise ValueError("index cannot be negative")
        object.__setattr__(self, "kind", SwingKind(self.kind))
        object.__setattr__(self, "price", _positive(self.price, "price"))
        object.__setattr__(self, "bar_time", _utc(self.bar_time, "bar_time"))
        object.__setattr__(self, "confirmed_at", _utc(self.confirmed_at, "confirmed_at"))
        if self.confirmed_at < self.bar_time:
            raise ValueError("confirmed_at cannot precede bar_time")


@dataclass(frozen=True, slots=True)
class StructureEvent:
    """A close-confirmed BOS or CHoCH through a previously confirmed swing."""

    symbol: str
    timeframe: Timeframe
    index: int
    kind: StructureEventKind
    direction: StructureDirection
    level: float
    close_price: float
    event_time: datetime
    broken_swing: SwingPoint

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if symbol != self.broken_swing.symbol:
            raise ValueError("Structure event and broken swing symbols must match")
        object.__setattr__(self, "symbol", symbol)
        timeframe = Timeframe.parse(self.timeframe)
        if timeframe is not self.broken_swing.timeframe:
            raise ValueError("Structure event and broken swing timeframes must match")
        object.__setattr__(self, "timeframe", timeframe)
        if self.index <= self.broken_swing.index:
            raise ValueError("Structure event must occur after its broken swing")
        object.__setattr__(self, "kind", StructureEventKind(self.kind))
        direction = StructureDirection(self.direction)
        if direction is StructureDirection.NEUTRAL:
            raise ValueError("Structure events must be directional")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "level", _positive(self.level, "level"))
        object.__setattr__(self, "close_price", _positive(self.close_price, "close_price"))
        object.__setattr__(self, "event_time", _utc(self.event_time, "event_time"))


@dataclass(frozen=True, slots=True)
class DisplacementCandle:
    """Large directional candle with a close near its range extreme."""

    symbol: str
    timeframe: Timeframe
    index: int
    direction: StructureDirection
    event_time: datetime
    body_size: float
    range_size: float
    body_atr: float
    close_location: float

    def __post_init__(self) -> None:
        direction = StructureDirection(self.direction)
        if direction is StructureDirection.NEUTRAL:
            raise ValueError("Displacement candles must be directional")
        object.__setattr__(self, "direction", direction)
        if self.index < 0:
            raise ValueError("index cannot be negative")
        object.__setattr__(self, "body_size", _non_negative(self.body_size, "body_size"))
        object.__setattr__(self, "range_size", _positive(self.range_size, "range_size"))
        object.__setattr__(self, "body_atr", _non_negative(self.body_atr, "body_atr"))
        location = float(self.close_location)
        if not isfinite(location) or not 0.0 <= location <= 1.0:
            raise ValueError("close_location must be between 0 and 1")
        object.__setattr__(self, "close_location", location)
        object.__setattr__(self, "event_time", _utc(self.event_time, "event_time"))


@dataclass(frozen=True, slots=True)
class MarketStructureConfig:
    """Swing confirmation and structural-break thresholds."""

    left_bars: int = 2
    right_bars: int = 2
    break_buffer_atr: float = 0.05
    displacement_body_atr: float = 1.0
    displacement_close_fraction: float = 0.70
    dealing_range_lookback_swings: int = 12

    def __post_init__(self) -> None:
        if self.left_bars < 1 or self.right_bars < 1:
            raise ValueError("left_bars and right_bars must be positive")
        object.__setattr__(
            self,
            "break_buffer_atr",
            _non_negative(self.break_buffer_atr, "break_buffer_atr"),
        )
        object.__setattr__(
            self,
            "displacement_body_atr",
            _positive(self.displacement_body_atr, "displacement_body_atr"),
        )
        close_fraction = float(self.displacement_close_fraction)
        if not isfinite(close_fraction) or not 0.5 <= close_fraction <= 1.0:
            raise ValueError("displacement_close_fraction must be between 0.5 and 1")
        object.__setattr__(self, "displacement_close_fraction", close_fraction)
        if self.dealing_range_lookback_swings < 2:
            raise ValueError("dealing_range_lookback_swings must be at least 2")


@dataclass(frozen=True, slots=True)
class MarketStructureState:
    """Point-in-time structure assessment for one market series."""

    symbol: str
    timeframe: Timeframe
    as_of: datetime
    bias: StructureDirection
    swings: tuple[SwingPoint, ...]
    events: tuple[StructureEvent, ...]
    displacements: tuple[DisplacementCandle, ...]
    range_low: float | None
    range_high: float | None
    equilibrium: float | None
    price_zone: DealingRangeZone

    @property
    def latest_event(self) -> StructureEvent | None:
        return self.events[-1] if self.events else None

    @property
    def latest_displacement(self) -> DisplacementCandle | None:
        return self.displacements[-1] if self.displacements else None


@dataclass(frozen=True, slots=True)
class StructureConfluence:
    """Directional agreement across requested timeframes."""

    symbol: str
    as_of: datetime
    primary_timeframe: Timeframe
    primary_bias: StructureDirection
    dominant_bias: StructureDirection
    alignment_score: float
    assessments: tuple[tuple[Timeframe, StructureDirection], ...]

    def __post_init__(self) -> None:
        score = float(self.alignment_score)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("alignment_score must be between 0 and 1")
        object.__setattr__(self, "alignment_score", score)


def detect_swings(
    series: MarketSeries,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
) -> tuple[SwingPoint, ...]:
    """Return pivots confirmed without using data beyond each confirmation bar."""

    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be positive")
    bars = series.bars
    if len(bars) < left_bars + right_bars + 1:
        return ()

    swings: list[SwingPoint] = []
    for index in range(left_bars, len(bars) - right_bars):
        bar = bars[index]
        left = bars[index - left_bars : index]
        right = bars[index + 1 : index + right_bars + 1]
        high = bar.mid.high
        low = bar.mid.low
        if high > max(item.mid.high for item in left) and high >= max(
            item.mid.high for item in right
        ):
            swings.append(
                SwingPoint(
                    symbol=series.symbol,
                    timeframe=series.timeframe,
                    index=index,
                    kind=SwingKind.HIGH,
                    price=high,
                    bar_time=bar.open_time,
                    confirmed_at=right[-1].close_time,
                )
            )
        if low < min(item.mid.low for item in left) and low <= min(
            item.mid.low for item in right
        ):
            swings.append(
                SwingPoint(
                    symbol=series.symbol,
                    timeframe=series.timeframe,
                    index=index,
                    kind=SwingKind.LOW,
                    price=low,
                    bar_time=bar.open_time,
                    confirmed_at=right[-1].close_time,
                )
            )
    return tuple(sorted(swings, key=lambda item: (item.index, item.kind.value)))


def detect_structure_events(
    series: MarketSeries,
    swings: tuple[SwingPoint, ...],
    *,
    break_buffer: float = 0.0,
) -> tuple[StructureEvent, ...]:
    """Detect close-confirmed BOS/CHoCH events through confirmed swing levels."""

    buffer = _non_negative(break_buffer, "break_buffer")
    events: list[StructureEvent] = []
    bias = StructureDirection.NEUTRAL
    broken: set[tuple[SwingKind, int]] = set()

    for index, bar in enumerate(series.bars):
        available = tuple(
            swing
            for swing in swings
            if swing.index < index
            and swing.confirmed_at <= bar.close_time
            and (swing.kind, swing.index) not in broken
        )
        highs = [item for item in available if item.kind is SwingKind.HIGH]
        lows = [item for item in available if item.kind is SwingKind.LOW]
        close = bar.mid.close

        candidate: tuple[StructureDirection, SwingPoint] | None = None
        if highs:
            high = max(highs, key=lambda item: item.index)
            if close > high.price + buffer:
                candidate = (StructureDirection.BULLISH, high)
        if lows:
            low = max(lows, key=lambda item: item.index)
            if close < low.price - buffer:
                candidate = (StructureDirection.BEARISH, low)

        if candidate is None:
            continue
        direction, swing = candidate
        kind = (
            StructureEventKind.CHANGE_OF_CHARACTER
            if bias not in {StructureDirection.NEUTRAL, direction}
            else StructureEventKind.BREAK_OF_STRUCTURE
        )
        event = StructureEvent(
            symbol=series.symbol,
            timeframe=series.timeframe,
            index=index,
            kind=kind,
            direction=direction,
            level=swing.price,
            close_price=close,
            event_time=bar.close_time,
            broken_swing=swing,
        )
        events.append(event)
        broken.add((swing.kind, swing.index))
        bias = direction
    return tuple(events)


def detect_displacements(
    series: MarketSeries,
    *,
    atr: float,
    body_atr_threshold: float = 1.0,
    close_fraction: float = 0.70,
) -> tuple[DisplacementCandle, ...]:
    """Detect large-bodied candles closing near the directional extreme."""

    volatility = _positive(atr, "atr")
    threshold = _positive(body_atr_threshold, "body_atr_threshold")
    if not isfinite(close_fraction) or not 0.5 <= close_fraction <= 1.0:
        raise ValueError("close_fraction must be between 0.5 and 1")

    result: list[DisplacementCandle] = []
    for index, bar in enumerate(series.bars):
        candle = bar.mid
        width = candle.high - candle.low
        if width <= 0.0:
            continue
        body = abs(candle.close - candle.open)
        body_atr = body / volatility
        if body_atr < threshold:
            continue
        bullish_location = (candle.close - candle.low) / width
        bearish_location = (candle.high - candle.close) / width
        direction: StructureDirection | None = None
        location = 0.0
        if candle.close > candle.open and bullish_location >= close_fraction:
            direction = StructureDirection.BULLISH
            location = bullish_location
        elif candle.close < candle.open and bearish_location >= close_fraction:
            direction = StructureDirection.BEARISH
            location = bearish_location
        if direction is not None:
            result.append(
                DisplacementCandle(
                    symbol=series.symbol,
                    timeframe=series.timeframe,
                    index=index,
                    direction=direction,
                    event_time=bar.close_time,
                    body_size=body,
                    range_size=width,
                    body_atr=body_atr,
                    close_location=location,
                )
            )
    return tuple(result)


def analyze_market_structure(
    series: MarketSeries,
    *,
    atr: float,
    config: MarketStructureConfig | None = None,
) -> MarketStructureState:
    """Build a complete point-in-time structure state from one market series."""

    settings = config or MarketStructureConfig()
    swings = detect_swings(
        series,
        left_bars=settings.left_bars,
        right_bars=settings.right_bars,
    )
    events = detect_structure_events(
        series,
        swings,
        break_buffer=atr * settings.break_buffer_atr,
    )
    displacements = detect_displacements(
        series,
        atr=atr,
        body_atr_threshold=settings.displacement_body_atr,
        close_fraction=settings.displacement_close_fraction,
    )
    bias = events[-1].direction if events else StructureDirection.NEUTRAL
    selected = swings[-settings.dealing_range_lookback_swings :]
    highs = [item.price for item in selected if item.kind is SwingKind.HIGH]
    lows = [item.price for item in selected if item.kind is SwingKind.LOW]
    range_high = max(highs) if highs else None
    range_low = min(lows) if lows else None
    equilibrium: float | None = None
    zone = DealingRangeZone.UNKNOWN
    if range_high is not None and range_low is not None and range_high > range_low:
        equilibrium = (range_high + range_low) / 2.0
        close = series.latest.mid.close
        epsilon = max((range_high - range_low) * 1e-9, 1e-12)
        if close > equilibrium + epsilon:
            zone = DealingRangeZone.PREMIUM
        elif close < equilibrium - epsilon:
            zone = DealingRangeZone.DISCOUNT
        else:
            zone = DealingRangeZone.EQUILIBRIUM

    return MarketStructureState(
        symbol=series.symbol,
        timeframe=series.timeframe,
        as_of=series.latest.close_time,
        bias=bias,
        swings=swings,
        events=events,
        displacements=displacements,
        range_low=range_low,
        range_high=range_high,
        equilibrium=equilibrium,
        price_zone=zone,
    )


def structure_confluence(
    context: MultiTimeframeContext,
    *,
    atr_by_timeframe: dict[Timeframe, float],
    config: MarketStructureConfig | None = None,
    timeframes: tuple[Timeframe, ...] | None = None,
) -> StructureConfluence:
    """Summarize directional structure agreement across selected timeframes."""

    selected = context.timeframes if timeframes is None else timeframes
    if not selected:
        raise ValueError("At least one timeframe is required")
    assessments: list[tuple[Timeframe, StructureDirection]] = []
    for timeframe in selected:
        parsed = Timeframe.parse(timeframe)
        try:
            atr = atr_by_timeframe[parsed]
        except KeyError as exc:
            raise KeyError(f"Missing ATR for timeframe {parsed.value}") from exc
        state = analyze_market_structure(context.get(parsed), atr=atr, config=config)
        assessments.append((parsed, state.bias))

    counts: dict[StructureDirection, int] = {}
    for _, bias in assessments:
        counts[bias] = counts.get(bias, 0) + 1
    dominant = max(counts, key=lambda item: counts[item])
    primary = next(
        bias for timeframe, bias in assessments if timeframe is context.primary_timeframe
    )
    return StructureConfluence(
        symbol=context.symbol,
        as_of=context.as_of,
        primary_timeframe=context.primary_timeframe,
        primary_bias=primary,
        dominant_bias=dominant,
        alignment_score=counts[dominant] / len(assessments),
        assessments=tuple(assessments),
    )
