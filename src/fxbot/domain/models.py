"""Immutable market-data domain models.

The data layer normalizes all timestamps to UTC and retains bid and ask prices.
That design prevents a later backtest from accidentally treating mid-price bars
as executable prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import TypeAlias

from fxbot.domain.enums import DataKind, Timeframe


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _positive_finite(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _optional_non_negative(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Trading-instrument metadata required for price and risk normalization."""

    symbol: str
    base_currency: str
    quote_currency: str
    digits: int
    point_size: float
    pip_size: float
    contract_size: float = 100_000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "base_currency", self.base_currency.strip().upper())
        object.__setattr__(self, "quote_currency", self.quote_currency.strip().upper())
        object.__setattr__(self, "point_size", _positive_finite(self.point_size, "point_size"))
        object.__setattr__(self, "pip_size", _positive_finite(self.pip_size, "pip_size"))
        object.__setattr__(
            self,
            "contract_size",
            _positive_finite(self.contract_size, "contract_size"),
        )
        if not self.base_currency or not self.quote_currency:
            raise ValueError("base_currency and quote_currency cannot be empty")
        if self.digits < 0:
            raise ValueError("digits must be non-negative")
        if self.pip_size < self.point_size:
            raise ValueError("pip_size cannot be smaller than point_size")


@dataclass(frozen=True, slots=True)
class Tick:
    """Executable two-sided quote observed at an instant."""

    symbol: str
    event_time: datetime
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None
    source: str = "unknown"
    sequence: int | None = None
    received_time: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "event_time", _utc(self.event_time, "event_time"))
        object.__setattr__(self, "bid", _positive_finite(self.bid, "bid"))
        object.__setattr__(self, "ask", _positive_finite(self.ask, "ask"))
        object.__setattr__(
            self,
            "bid_size",
            _optional_non_negative(self.bid_size, "bid_size"),
        )
        object.__setattr__(
            self,
            "ask_size",
            _optional_non_negative(self.ask_size, "ask_size"),
        )
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.received_time is not None:
            object.__setattr__(
                self,
                "received_time",
                _utc(self.received_time, "received_time"),
            )
        object.__setattr__(self, "source", self.source.strip() or "unknown")

    @property
    def kind(self) -> DataKind:
        return DataKind.TICK

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def latency(self) -> timedelta | None:
        if self.received_time is None:
            return None
        return self.received_time - self.event_time

    def spread_pips(self, spec: SymbolSpec) -> float:
        if spec.symbol != self.symbol:
            raise ValueError(f"SymbolSpec {spec.symbol} does not match tick {self.symbol}")
        return self.spread / spec.pip_size


@dataclass(frozen=True, slots=True)
class OHLC:
    """One side of a quote bar: bid, ask, or derived mid."""

    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, low, and close")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, high, and close")


@dataclass(frozen=True, slots=True)
class Bar:
    """Completed or in-progress two-sided OHLC bar."""

    symbol: str
    open_time: datetime
    timeframe: Timeframe
    bid: OHLC
    ask: OHLC
    mid_ohlc: OHLC | None = None
    tick_volume: int = 0
    real_volume: float | None = None
    source: str = "unknown"
    complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "open_time", _utc(self.open_time, "open_time"))
        object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe))
        if self.timeframe is Timeframe.TICK:
            raise ValueError("Bar timeframe cannot be tick")
        if self.tick_volume < 0:
            raise ValueError("tick_volume must be non-negative")
        object.__setattr__(
            self,
            "real_volume",
            _optional_non_negative(self.real_volume, "real_volume"),
        )
        if self.ask.open < self.bid.open or self.ask.close < self.bid.close:
            raise ValueError("ask open/close cannot be below bid open/close")
        if self.ask.high < self.bid.high or self.ask.low < self.bid.low:
            raise ValueError("ask high/low cannot be below bid high/low")
        if self.mid_ohlc is not None:
            for field_name in ("open", "high", "low", "close"):
                bid_value = getattr(self.bid, field_name)
                mid_value = getattr(self.mid_ohlc, field_name)
                ask_value = getattr(self.ask, field_name)
                if not bid_value <= mid_value <= ask_value:
                    raise ValueError(
                        f"mid_ohlc.{field_name} must be between bid and ask values"
                    )
        object.__setattr__(self, "source", self.source.strip() or "unknown")

    @property
    def kind(self) -> DataKind:
        return DataKind.BAR

    @property
    def close_time(self) -> datetime:
        seconds = self.timeframe.seconds
        if seconds is None:  # defensive; barred by __post_init__
            raise RuntimeError("Tick timeframe does not have a close time")
        return self.open_time + timedelta(seconds=seconds)

    @property
    def mid(self) -> OHLC:
        if self.mid_ohlc is not None:
            return self.mid_ohlc
        return OHLC(
            open=(self.bid.open + self.ask.open) / 2.0,
            high=(self.bid.high + self.ask.high) / 2.0,
            low=(self.bid.low + self.ask.low) / 2.0,
            close=(self.bid.close + self.ask.close) / 2.0,
        )

    @property
    def spread_open(self) -> float:
        return self.ask.open - self.bid.open

    @property
    def spread_close(self) -> float:
        return self.ask.close - self.bid.close


MarketDataRecord: TypeAlias = Tick | Bar


@dataclass(frozen=True, slots=True)
class HistoricalDataRequest:
    """Half-open time-range query: ``start <= event_time < end``."""

    symbol: str
    kind: DataKind
    start: datetime | None = None
    end: datetime | None = None
    timeframe: Timeframe | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "kind", DataKind(self.kind))
        if self.start is not None:
            object.__setattr__(self, "start", _utc(self.start, "start"))
        if self.end is not None:
            object.__setattr__(self, "end", _utc(self.end, "end"))
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start must be earlier than end")

        if self.kind is DataKind.TICK:
            if self.timeframe not in (None, Timeframe.TICK):
                raise ValueError("Tick requests cannot specify a bar timeframe")
            object.__setattr__(self, "timeframe", Timeframe.TICK)
        else:
            if self.timeframe in (None, Timeframe.TICK):
                raise ValueError("Bar requests require a non-tick timeframe")
            object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe))

    def contains(self, timestamp: datetime) -> bool:
        value = _utc(timestamp, "timestamp")
        return (self.start is None or value >= self.start) and (
            self.end is None or value < self.end
        )


@dataclass(frozen=True, slots=True)
class LiveSubscription:
    """Subscription filter used by broker and queue-based live adapters."""

    symbols: frozenset[str]
    timeframes: frozenset[Timeframe] = frozenset({Timeframe.TICK})

    def __post_init__(self) -> None:
        normalized_symbols = frozenset(_symbol(item) for item in self.symbols)
        normalized_timeframes = frozenset(Timeframe.parse(item) for item in self.timeframes)
        if not normalized_symbols:
            raise ValueError("At least one symbol is required")
        if not normalized_timeframes:
            raise ValueError("At least one timeframe is required")
        object.__setattr__(self, "symbols", normalized_symbols)
        object.__setattr__(self, "timeframes", normalized_timeframes)

    def accepts(self, record: MarketDataRecord) -> bool:
        if record.symbol not in self.symbols:
            return False
        if isinstance(record, Tick):
            return Timeframe.TICK in self.timeframes
        return record.timeframe in self.timeframes
