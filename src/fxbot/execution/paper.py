"""Deterministic bid/ask-aware paper broker for production dry runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite

from fxbot.execution.broker import OrderNotFoundError, PermanentBrokerError
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionFill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    TimeInForce,
)


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Execution assumptions for deterministic paper trading."""

    max_fill_quantity_per_quote: float | None = None
    commission_per_unit: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        if self.max_fill_quantity_per_quote is not None:
            value = float(self.max_fill_quantity_per_quote)
            if not isfinite(value) or value <= 0.0:
                raise ValueError("max_fill_quantity_per_quote must be positive and finite")
            object.__setattr__(self, "max_fill_quantity_per_quote", value)
        for name in ("commission_per_unit", "slippage"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


class PaperBroker:
    """In-memory broker supporting market, limit, stop, and stop-limit orders."""

    def __init__(self, config: PaperBrokerConfig | None = None) -> None:
        self.config = config or PaperBrokerConfig()
        self._orders: dict[str, BrokerOrder] = {}
        self._intents: dict[str, OrderIntent] = {}
        self._client_to_broker: dict[str, str] = {}
        self._quotes: dict[str, Quote] = {}
        self._fills: list[ExecutionFill] = []
        self._triggered_stop_limits: set[str] = set()
        self._order_sequence = 0
        self._fill_sequence = 0

    @property
    def name(self) -> str:
        return "paper"

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        existing = self.find_order_by_client_id(intent.client_order_id)
        if existing is not None:
            return existing

        broker_order_id = f"PAPER-{self._order_sequence:010d}"
        self._order_sequence += 1
        order = BrokerOrder(
            broker_order_id=broker_order_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status=OrderStatus.ACKNOWLEDGED,
            requested_quantity=intent.quantity,
            filled_quantity=0.0,
            average_fill_price=None,
            submitted_at=intent.created_at,
            updated_at=intent.created_at,
            metadata=intent.metadata,
        )
        self._orders[broker_order_id] = order
        self._intents[broker_order_id] = intent
        self._client_to_broker[intent.client_order_id] = broker_order_id

        quote = self._quotes.get(intent.symbol)
        if intent.order_type is OrderType.MARKET and quote is None:
            rejected = replace(
                order,
                status=OrderStatus.REJECTED,
                rejection_reason="No executable quote available",
            )
            self._orders[broker_order_id] = rejected
            raise PermanentBrokerError("No executable quote available")

        if quote is not None:
            self._try_execute(broker_order_id, quote, initial_attempt=True)
        elif intent.time_in_force in {TimeInForce.IOC, TimeInForce.FOK}:
            self._orders[broker_order_id] = replace(
                order,
                status=OrderStatus.CANCELLED,
                updated_at=intent.created_at,
            )
        return self._orders[broker_order_id]

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        order = self.get_order(broker_order_id)
        if order.status.terminal:
            return order
        cancelled = replace(
            order,
            status=OrderStatus.CANCELLED,
            updated_at=self._latest_time(order.symbol, order.updated_at),
        )
        self._orders[broker_order_id] = cancelled
        return cancelled

    def get_order(self, broker_order_id: str) -> BrokerOrder:
        try:
            return self._orders[broker_order_id]
        except KeyError as exc:
            raise OrderNotFoundError(f"Unknown broker order: {broker_order_id}") from exc

    def find_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        broker_id = self._client_to_broker.get(client_order_id)
        return self._orders.get(broker_id) if broker_id is not None else None

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        return tuple(
            sorted(
                (order for order in self._orders.values() if order.status.active),
                key=lambda item: item.broker_order_id,
            )
        )

    def drain_fills(self) -> tuple[ExecutionFill, ...]:
        output = tuple(self._fills)
        self._fills.clear()
        return output

    def update_quote(self, quote: Quote) -> tuple[ExecutionFill, ...]:
        current = self._quotes.get(quote.symbol)
        if current is not None and quote.timestamp < current.timestamp:
            raise ValueError("Quotes must be chronological per symbol")
        self._quotes[quote.symbol] = quote
        before = len(self._fills)
        for broker_order_id in tuple(sorted(self._orders)):
            order = self._orders[broker_order_id]
            if order.symbol != quote.symbol or not order.status.active:
                continue
            intent = self._intents[broker_order_id]
            if (
                intent.time_in_force is TimeInForce.DAY
                and quote.timestamp.date() > intent.created_at.date()
            ):
                self._orders[broker_order_id] = replace(
                    order,
                    status=OrderStatus.EXPIRED,
                    updated_at=quote.timestamp,
                )
                continue
            self._try_execute(broker_order_id, quote, initial_attempt=False)
        return tuple(self._fills[before:])

    @property
    def orders(self) -> tuple[BrokerOrder, ...]:
        return tuple(self._orders[key] for key in sorted(self._orders))

    def _try_execute(
        self,
        broker_order_id: str,
        quote: Quote,
        *,
        initial_attempt: bool,
    ) -> None:
        order = self._orders[broker_order_id]
        intent = self._intents[broker_order_id]
        executable, price = self._execution_price(broker_order_id, intent, quote)
        if not executable or price is None:
            if initial_attempt and intent.time_in_force in {TimeInForce.IOC, TimeInForce.FOK}:
                self._orders[broker_order_id] = replace(
                    order,
                    status=OrderStatus.CANCELLED,
                    updated_at=quote.timestamp,
                )
            return

        remaining = order.remaining_quantity
        maximum = self.config.max_fill_quantity_per_quote
        fill_quantity = remaining if maximum is None else min(remaining, maximum)
        if intent.time_in_force is TimeInForce.FOK and fill_quantity + 1e-12 < remaining:
            self._orders[broker_order_id] = replace(
                order,
                status=OrderStatus.CANCELLED,
                updated_at=quote.timestamp,
            )
            return

        adjusted_price = price + self.config.slippage * intent.side.sign
        total_filled = order.filled_quantity + fill_quantity
        weighted_price = (
            adjusted_price
            if order.filled_quantity <= 0.0 or order.average_fill_price is None
            else (
                order.average_fill_price * order.filled_quantity
                + adjusted_price * fill_quantity
            )
            / total_filled
        )
        status = (
            OrderStatus.FILLED
            if abs(total_filled - order.requested_quantity) <= 1e-12
            else OrderStatus.PARTIALLY_FILLED
        )
        updated = replace(
            order,
            status=status,
            filled_quantity=total_filled,
            average_fill_price=weighted_price,
            updated_at=quote.timestamp,
        )
        self._orders[broker_order_id] = updated
        fill = ExecutionFill(
            execution_id=f"PAPER-FILL-{self._fill_sequence:010d}",
            broker_order_id=broker_order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=adjusted_price,
            executed_at=quote.timestamp,
            commission=fill_quantity * self.config.commission_per_unit,
            liquidity="paper",
        )
        self._fill_sequence += 1
        self._fills.append(fill)

        if intent.time_in_force is TimeInForce.IOC and status is not OrderStatus.FILLED:
            self._orders[broker_order_id] = replace(
                updated,
                status=OrderStatus.CANCELLED,
            )

    def _execution_price(
        self,
        broker_order_id: str,
        intent: OrderIntent,
        quote: Quote,
    ) -> tuple[bool, float | None]:
        executable_price = quote.ask if intent.side is OrderSide.BUY else quote.bid
        if intent.order_type is OrderType.MARKET:
            return True, executable_price
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            condition = (
                executable_price <= intent.limit_price
                if intent.side is OrderSide.BUY
                else executable_price >= intent.limit_price
            )
            return condition, executable_price if condition else None
        if intent.order_type is OrderType.STOP:
            assert intent.stop_price is not None
            condition = (
                executable_price >= intent.stop_price
                if intent.side is OrderSide.BUY
                else executable_price <= intent.stop_price
            )
            return condition, executable_price if condition else None

        assert intent.stop_price is not None
        assert intent.limit_price is not None
        triggered = broker_order_id in self._triggered_stop_limits
        if not triggered:
            triggered = (
                executable_price >= intent.stop_price
                if intent.side is OrderSide.BUY
                else executable_price <= intent.stop_price
            )
            if triggered:
                self._triggered_stop_limits.add(broker_order_id)
        if not triggered:
            return False, None
        limit_condition = (
            executable_price <= intent.limit_price
            if intent.side is OrderSide.BUY
            else executable_price >= intent.limit_price
        )
        return limit_condition, executable_price if limit_condition else None

    def _latest_time(self, symbol: str, fallback: datetime) -> datetime:
        quote = self._quotes.get(symbol)
        return quote.timestamp if quote is not None else fallback
