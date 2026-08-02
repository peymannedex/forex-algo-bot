"""Immutable event, order, and fill contracts for deterministic simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import TypeAlias

from fxbot.domain.models import Bar, Tick

MarketDataRecord: TypeAlias = Tick | Bar


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


def market_record_time(record: MarketDataRecord) -> datetime:
    """Return the first time at which the complete record is knowable."""

    if isinstance(record, Tick):
        return record.event_time
    return record.close_time if record.complete else record.open_time


class EventKind(StrEnum):
    """Machine-readable event types retained in the audit trail."""

    MARKET = "market"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    FILL = "fill"
    SWAP = "swap"
    EQUITY = "equity"


class OrderSide(StrEnum):
    """Executable order direction."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> float:
        return 1.0 if self is OrderSide.BUY else -1.0

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class OrderType(StrEnum):
    """Supported simulated order instructions."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(StrEnum):
    """Supported order lifetimes."""

    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(StrEnum):
    """Durable simulated order state."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """One chronologically ordered market observation."""

    sequence: int
    timestamp: datetime
    record: MarketDataRecord

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        timestamp = _utc(self.timestamp, "timestamp")
        expected = market_record_time(self.record)
        if timestamp != expected:
            raise ValueError("timestamp must equal the market record availability time")
        object.__setattr__(self, "timestamp", timestamp)

    @property
    def symbol(self) -> str:
        return self.record.symbol


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Strategy-generated order submitted to the simulated broker."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    volume: float
    submitted_at: datetime
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    client_tag: str = "strategy"
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _identifier(self.order_id, "order_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "side", OrderSide(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(self, "submitted_at", _utc(self.submitted_at, "submitted_at"))
        object.__setattr__(self, "time_in_force", TimeInForce(self.time_in_force))
        object.__setattr__(self, "client_tag", self.client_tag.strip() or "strategy")

        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", _positive(self.limit_price, "limit_price"))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _positive(self.stop_price, "stop_price"))
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP orders require stop_price")
        if self.order_type is OrderType.MARKET and (
            self.limit_price is not None or self.stop_price is not None
        ):
            raise ValueError("MARKET orders cannot define limit_price or stop_price")
        if self.order_type is OrderType.LIMIT and self.stop_price is not None:
            raise ValueError("LIMIT orders cannot define stop_price")
        if self.order_type is OrderType.STOP and self.limit_price is not None:
            raise ValueError("STOP orders cannot define limit_price")

        normalized_metadata: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_key, raw_value in self.metadata:
            key = _identifier(raw_key, "metadata key")
            if key in seen:
                raise ValueError(f"Duplicate metadata key: {key}")
            seen.add(key)
            normalized_metadata.append((key, str(raw_value)))
        object.__setattr__(self, "metadata", tuple(sorted(normalized_metadata)))


@dataclass(frozen=True, slots=True)
class OrderState:
    """Immutable broker-side state of one submitted order."""

    request: OrderRequest
    status: OrderStatus
    remaining_volume: float
    filled_volume: float
    average_fill_price: float | None
    activated_after_sequence: int
    updated_at: datetime
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", OrderStatus(self.status))
        object.__setattr__(
            self,
            "remaining_volume",
            _non_negative(self.remaining_volume, "remaining_volume"),
        )
        object.__setattr__(
            self,
            "filled_volume",
            _non_negative(self.filled_volume, "filled_volume"),
        )
        if self.average_fill_price is not None:
            object.__setattr__(
                self,
                "average_fill_price",
                _positive(self.average_fill_price, "average_fill_price"),
            )
        if self.activated_after_sequence < 0:
            raise ValueError("activated_after_sequence must be non-negative")
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        if abs(self.remaining_volume + self.filled_volume - self.request.volume) > 1e-9:
            raise ValueError("remaining_volume plus filled_volume must equal requested volume")
        if self.status is OrderStatus.FILLED and self.remaining_volume > 1e-12:
            raise ValueError("FILLED orders require zero remaining_volume")
        if self.status is OrderStatus.PARTIALLY_FILLED and (
            self.filled_volume <= 0.0 or self.remaining_volume <= 0.0
        ):
            raise ValueError("PARTIALLY_FILLED orders require filled and remaining volume")
        if self.status is OrderStatus.REJECTED and not self.rejection_reason:
            raise ValueError("REJECTED orders require rejection_reason")


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    """One deterministic simulated broker execution."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    volume: float
    price: float
    timestamp: datetime
    sequence: int
    commission: float = 0.0
    slippage_amount: float = 0.0
    liquidity: str = "simulated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_id", _identifier(self.fill_id, "fill_id"))
        object.__setattr__(self, "order_id", _identifier(self.order_id, "order_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "side", OrderSide(self.side))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(self, "price", _positive(self.price, "price"))
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "timestamp"))
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "commission", _non_negative(self.commission, "commission"))
        object.__setattr__(
            self,
            "slippage_amount",
            _non_negative(self.slippage_amount, "slippage_amount"),
        )
        object.__setattr__(self, "liquidity", self.liquidity.strip() or "simulated")


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Compact immutable record of one backtest state transition."""

    sequence: int
    timestamp: datetime
    kind: EventKind
    message: str
    order_id: str | None = None
    fill_id: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "timestamp"))
        object.__setattr__(self, "kind", EventKind(self.kind))
        message = self.message.strip()
        if not message:
            raise ValueError("message cannot be empty")
        object.__setattr__(self, "message", message)
        if self.order_id is not None:
            object.__setattr__(self, "order_id", _identifier(self.order_id, "order_id"))
        if self.fill_id is not None:
            object.__setattr__(self, "fill_id", _identifier(self.fill_id, "fill_id"))


BacktestEvent: TypeAlias = MarketEvent | AuditEvent
