"""MetaTrader 5 request mapping and broker-constraint normalization."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from hashlib import sha256
from math import isfinite
from typing import Any, Protocol

from fxbot.execution.models import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    TimeInForce,
)


class MT5ConstantSource(Protocol):
    """Structural source of MetaTrader 5 integer constants."""


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _constant(client: MT5ConstantSource, name: str, default: int | None = None) -> int:
    value = getattr(client, name, default)
    if value is None:
        raise ValueError(f"MT5 client does not expose required constant {name}")
    return int(value)


def _decimal(value: float) -> Decimal:
    if not isfinite(float(value)):
        raise ValueError("numeric value must be finite")
    return Decimal(str(float(value)))


@dataclass(frozen=True, slots=True)
class MT5SymbolSpec:
    """Execution-relevant subset of MT5 symbol metadata."""

    symbol: str
    digits: int
    point: float
    tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int = 0
    freeze_level_points: int = 0
    filling_mode: int | None = None

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        object.__setattr__(self, "symbol", symbol)
        if self.digits < 0:
            raise ValueError("digits must be non-negative")
        for name in ("point", "tick_size", "volume_min", "volume_max", "volume_step"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.volume_max < self.volume_min:
            raise ValueError("volume_max cannot be below volume_min")
        if self.stops_level_points < 0 or self.freeze_level_points < 0:
            raise ValueError("stop and freeze levels must be non-negative")

    @classmethod
    def from_mt5(cls, symbol: str, info: Any) -> MT5SymbolSpec:
        point = float(_field(info, "point"))
        tick_size = float(_field(info, "trade_tick_size", point)) or point
        return cls(
            symbol=symbol,
            digits=int(_field(info, "digits", 5)),
            point=point,
            tick_size=tick_size,
            volume_min=float(_field(info, "volume_min")),
            volume_max=float(_field(info, "volume_max")),
            volume_step=float(_field(info, "volume_step")),
            stops_level_points=int(_field(info, "trade_stops_level", 0)),
            freeze_level_points=int(_field(info, "trade_freeze_level", 0)),
            filling_mode=(
                int(_field(info, "filling_mode"))
                if _field(info, "filling_mode", None) is not None
                else None
            ),
        )

    @property
    def stops_distance(self) -> float:
        return self.stops_level_points * self.point

    @property
    def freeze_distance(self) -> float:
        return self.freeze_level_points * self.point


def normalize_volume(quantity: float, spec: MT5SymbolSpec) -> float:
    """Round volume down to the broker step without increasing risk."""

    requested = _decimal(quantity)
    minimum = _decimal(spec.volume_min)
    maximum = _decimal(spec.volume_max)
    step = _decimal(spec.volume_step)
    if requested < minimum:
        raise ValueError(
            f"Requested volume {quantity} is below broker minimum {spec.volume_min}"
        )
    bounded = min(requested, maximum)
    steps = (bounded / step).to_integral_value(rounding=ROUND_FLOOR)
    normalized = steps * step
    if normalized < minimum:
        raise ValueError("Normalized volume is below broker minimum")
    return float(normalized)


def normalize_price(price: float, spec: MT5SymbolSpec) -> float:
    """Round a price to the symbol tick size and display precision."""

    value = _decimal(price)
    tick = _decimal(spec.tick_size)
    ticks = (value / tick).to_integral_value(rounding=ROUND_HALF_UP)
    normalized = ticks * tick
    return round(float(normalized), spec.digits)


def client_order_comment(client_order_id: str, *, maximum_length: int = 31) -> str:
    """Return a deterministic MT5-safe comment used for recovery searches."""

    raw = client_order_id.strip()
    if not raw:
        raise ValueError("client_order_id cannot be empty")
    digest = sha256(raw.encode()).hexdigest()[:10]
    slug = "".join(character for character in raw if character.isalnum() or character in "-_")
    prefix = f"fxb:{digest}:"
    room = maximum_length - len(prefix)
    if room < 1:
        raise ValueError("maximum_length is too small")
    return prefix + (slug[:room] or "order")


def _order_type_constant(client: MT5ConstantSource, intent: OrderIntent) -> int:
    names = {
        (OrderSide.BUY, OrderType.MARKET): "ORDER_TYPE_BUY",
        (OrderSide.SELL, OrderType.MARKET): "ORDER_TYPE_SELL",
        (OrderSide.BUY, OrderType.LIMIT): "ORDER_TYPE_BUY_LIMIT",
        (OrderSide.SELL, OrderType.LIMIT): "ORDER_TYPE_SELL_LIMIT",
        (OrderSide.BUY, OrderType.STOP): "ORDER_TYPE_BUY_STOP",
        (OrderSide.SELL, OrderType.STOP): "ORDER_TYPE_SELL_STOP",
        (OrderSide.BUY, OrderType.STOP_LIMIT): "ORDER_TYPE_BUY_STOP_LIMIT",
        (OrderSide.SELL, OrderType.STOP_LIMIT): "ORDER_TYPE_SELL_STOP_LIMIT",
    }
    return _constant(client, names[(intent.side, intent.order_type)])


def _time_and_filling(
    client: MT5ConstantSource,
    intent: OrderIntent,
    spec: MT5SymbolSpec,
) -> tuple[int, int]:
    if intent.time_in_force is TimeInForce.DAY:
        type_time = _constant(client, "ORDER_TIME_DAY")
    else:
        type_time = _constant(client, "ORDER_TIME_GTC")

    if intent.time_in_force is TimeInForce.IOC:
        filling = _constant(client, "ORDER_FILLING_IOC")
    elif intent.time_in_force is TimeInForce.FOK:
        filling = _constant(client, "ORDER_FILLING_FOK")
    elif intent.order_type is not OrderType.MARKET or spec.filling_mode is None:
        filling = _constant(client, "ORDER_FILLING_RETURN")
    else:
        symbol_ioc = getattr(client, "SYMBOL_FILLING_IOC", None)
        symbol_fok = getattr(client, "SYMBOL_FILLING_FOK", None)
        if symbol_ioc is not None and spec.filling_mode & int(symbol_ioc):
            filling = _constant(client, "ORDER_FILLING_IOC")
        elif symbol_fok is not None and spec.filling_mode & int(symbol_fok):
            filling = _constant(client, "ORDER_FILLING_FOK")
        else:
            filling = _constant(client, "ORDER_FILLING_RETURN")
    return type_time, filling


def validate_entry_prices(intent: OrderIntent, quote: Quote, spec: MT5SymbolSpec) -> None:
    """Validate pending prices against the current quote and broker stop level."""

    distance = spec.stops_distance
    tolerance = max(spec.tick_size, spec.point) * 0.5
    if intent.order_type is OrderType.LIMIT:
        assert intent.limit_price is not None
        if intent.side is OrderSide.BUY and intent.limit_price > quote.ask - distance + tolerance:
            raise ValueError("BUY_LIMIT must be below ask by the broker stops level")
        if intent.side is OrderSide.SELL and intent.limit_price < quote.bid + distance - tolerance:
            raise ValueError("SELL_LIMIT must be above bid by the broker stops level")
    elif intent.order_type is OrderType.STOP:
        assert intent.stop_price is not None
        if intent.side is OrderSide.BUY and intent.stop_price < quote.ask + distance - tolerance:
            raise ValueError("BUY_STOP must be above ask by the broker stops level")
        if intent.side is OrderSide.SELL and intent.stop_price > quote.bid - distance + tolerance:
            raise ValueError("SELL_STOP must be below bid by the broker stops level")
    elif intent.order_type is OrderType.STOP_LIMIT:
        assert intent.stop_price is not None and intent.limit_price is not None
        if intent.side is OrderSide.BUY:
            if intent.stop_price < quote.ask + distance - tolerance:
                raise ValueError("BUY_STOP_LIMIT trigger is too close to ask")
            if intent.limit_price > intent.stop_price + tolerance:
                raise ValueError("BUY_STOP_LIMIT limit cannot exceed its trigger")
        else:
            if intent.stop_price > quote.bid - distance + tolerance:
                raise ValueError("SELL_STOP_LIMIT trigger is too close to bid")
            if intent.limit_price < intent.stop_price - tolerance:
                raise ValueError("SELL_STOP_LIMIT limit cannot be below its trigger")


def validate_freeze_distance(price: float, quote: Quote, spec: MT5SymbolSpec) -> None:
    """Reject modification prices inside the broker freeze zone."""

    reference_distance = min(abs(price - quote.bid), abs(price - quote.ask))
    if reference_distance + 1e-12 < spec.freeze_distance:
        raise ValueError("Price is inside the broker freeze level")


def build_mt5_request(
    client: MT5ConstantSource,
    intent: OrderIntent,
    spec: MT5SymbolSpec,
    quote: Quote,
    *,
    magic_number: int,
    deviation_points: int,
) -> dict[str, Any]:
    """Map a normalized order intent to an ``order_send`` request."""

    if intent.symbol != spec.symbol or intent.symbol != quote.symbol:
        raise ValueError("Intent, symbol specification, and quote must use the same symbol")
    if magic_number <= 0:
        raise ValueError("magic_number must be positive")
    if deviation_points < 0:
        raise ValueError("deviation_points must be non-negative")

    validate_entry_prices(intent, quote, spec)
    type_time, type_filling = _time_and_filling(client, intent, spec)
    request: dict[str, Any] = {
        "action": _constant(
            client,
            "TRADE_ACTION_DEAL" if intent.order_type is OrderType.MARKET else "TRADE_ACTION_PENDING",
        ),
        "symbol": intent.symbol,
        "volume": normalize_volume(intent.quantity, spec),
        "type": _order_type_constant(client, intent),
        "deviation": deviation_points,
        "magic": magic_number,
        "comment": client_order_comment(intent.client_order_id),
        "type_time": type_time,
        "type_filling": type_filling,
    }
    if intent.order_type is OrderType.MARKET:
        request["price"] = normalize_price(
            quote.ask if intent.side is OrderSide.BUY else quote.bid,
            spec,
        )
    elif intent.order_type is OrderType.LIMIT:
        assert intent.limit_price is not None
        request["price"] = normalize_price(intent.limit_price, spec)
    elif intent.order_type is OrderType.STOP:
        assert intent.stop_price is not None
        request["price"] = normalize_price(intent.stop_price, spec)
    else:
        assert intent.stop_price is not None and intent.limit_price is not None
        request["price"] = normalize_price(intent.stop_price, spec)
        request["stoplimit"] = normalize_price(intent.limit_price, spec)
    return request


def side_from_mt5_order_type(client: MT5ConstantSource, raw_type: int) -> OrderSide:
    buy_types = {
        _constant(client, "ORDER_TYPE_BUY"),
        _constant(client, "ORDER_TYPE_BUY_LIMIT"),
        _constant(client, "ORDER_TYPE_BUY_STOP"),
        _constant(client, "ORDER_TYPE_BUY_STOP_LIMIT"),
    }
    return OrderSide.BUY if int(raw_type) in buy_types else OrderSide.SELL


def order_type_from_mt5(client: MT5ConstantSource, raw_type: int) -> OrderType:
    mapping = {
        _constant(client, "ORDER_TYPE_BUY"): OrderType.MARKET,
        _constant(client, "ORDER_TYPE_SELL"): OrderType.MARKET,
        _constant(client, "ORDER_TYPE_BUY_LIMIT"): OrderType.LIMIT,
        _constant(client, "ORDER_TYPE_SELL_LIMIT"): OrderType.LIMIT,
        _constant(client, "ORDER_TYPE_BUY_STOP"): OrderType.STOP,
        _constant(client, "ORDER_TYPE_SELL_STOP"): OrderType.STOP,
        _constant(client, "ORDER_TYPE_BUY_STOP_LIMIT"): OrderType.STOP_LIMIT,
        _constant(client, "ORDER_TYPE_SELL_STOP_LIMIT"): OrderType.STOP_LIMIT,
    }
    try:
        return mapping[int(raw_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported MT5 order type: {raw_type}") from exc


def status_from_mt5_state(client: MT5ConstantSource, raw_state: int) -> OrderStatus:
    mapping = {
        _constant(client, "ORDER_STATE_STARTED"): OrderStatus.SUBMITTED,
        _constant(client, "ORDER_STATE_PLACED"): OrderStatus.ACKNOWLEDGED,
        _constant(client, "ORDER_STATE_CANCELED"): OrderStatus.CANCELLED,
        _constant(client, "ORDER_STATE_PARTIAL"): OrderStatus.PARTIALLY_FILLED,
        _constant(client, "ORDER_STATE_FILLED"): OrderStatus.FILLED,
        _constant(client, "ORDER_STATE_REJECTED"): OrderStatus.REJECTED,
        _constant(client, "ORDER_STATE_EXPIRED"): OrderStatus.EXPIRED,
        _constant(client, "ORDER_STATE_REQUEST_ADD"): OrderStatus.SUBMITTED,
        _constant(client, "ORDER_STATE_REQUEST_MODIFY"): OrderStatus.ACKNOWLEDGED,
        _constant(client, "ORDER_STATE_REQUEST_CANCEL"): OrderStatus.CANCEL_PENDING,
    }
    return mapping.get(int(raw_state), OrderStatus.UNKNOWN)


__all__ = [
    "MT5ConstantSource",
    "MT5SymbolSpec",
    "build_mt5_request",
    "client_order_comment",
    "normalize_price",
    "normalize_volume",
    "order_type_from_mt5",
    "side_from_mt5_order_type",
    "status_from_mt5_state",
    "validate_entry_prices",
    "validate_freeze_distance",
]
