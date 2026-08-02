"""Execution-event consumption, deduplication, and state reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from fxbot.execution.broker import BrokerAdapter, FillSink, OrderSink, ensure_unique_fills
from fxbot.execution.lifecycle import InvalidOrderTransitionError, validate_transition
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionFill,
    OrderStatus,
)

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Summary of one broker synchronization pass."""

    new_fills: int
    duplicate_fills: int
    order_updates: int
    warnings: int


class ExecutionRuntime:
    """Consume broker updates exactly once and enforce local lifecycle monotonicity."""

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        fill_sinks: tuple[FillSink, ...] = (),
        order_sinks: tuple[OrderSink, ...] = (),
        clock: Clock | None = None,
    ) -> None:
        self.broker = broker
        self.fill_sinks = fill_sinks
        self.order_sinks = order_sinks
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._orders: dict[str, BrokerOrder] = {}
        self._seen_execution_ids: set[str] = set()
        self._events: list[ExecutionEvent] = []
        self._sequence = 0

    @property
    def audit_events(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._events)

    @property
    def known_orders(self) -> tuple[BrokerOrder, ...]:
        return tuple(self._orders[key] for key in sorted(self._orders))

    def observe_order(self, order: BrokerOrder) -> bool:
        """Store and publish an order update; return False for an exact duplicate."""

        current = self._orders.get(order.broker_order_id)
        if current == order:
            return False
        if current is not None:
            validate_transition(current, order)
        self._orders[order.broker_order_id] = order
        for sink in self.order_sinks:
            sink.on_order(order)
        kind = self._order_event_kind(order.status)
        if kind is not None:
            self._audit(
                kind,
                f"Order state synchronized: {order.status.value}",
                order=order,
            )
        return True

    def sync(self) -> SyncResult:
        """Poll current open orders and consume all newly available fills."""

        updates = 0
        warnings = 0
        for order in self.broker.list_open_orders():
            try:
                updates += int(self.observe_order(order))
            except InvalidOrderTransitionError as exc:
                warnings += 1
                self._audit(
                    ExecutionEventKind.RECONCILIATION_WARNING,
                    str(exc),
                    order=order,
                )

        fills = ensure_unique_fills(self.broker.drain_fills())
        new_fills = 0
        duplicate_fills = 0
        for fill in fills:
            if fill.execution_id in self._seen_execution_ids:
                duplicate_fills += 1
                self._audit(
                    ExecutionEventKind.DUPLICATE_SUPPRESSED,
                    "Duplicate execution report suppressed",
                    fill=fill,
                )
                continue
            self._seen_execution_ids.add(fill.execution_id)
            new_fills += 1
            for sink in self.fill_sinks:
                sink.on_fill(fill)
            self._audit(ExecutionEventKind.FILL, "Execution report consumed", fill=fill)
            try:
                latest = self.broker.get_order(fill.broker_order_id)
                updates += int(self.observe_order(latest))
            except InvalidOrderTransitionError as exc:
                warnings += 1
                self._audit(
                    ExecutionEventKind.RECONCILIATION_WARNING,
                    str(exc),
                    fill=fill,
                )

        return SyncResult(new_fills, duplicate_fills, updates, warnings)

    @staticmethod
    def _order_event_kind(status: OrderStatus) -> ExecutionEventKind | None:
        mapping = {
            OrderStatus.ACKNOWLEDGED: ExecutionEventKind.ORDER_ACKNOWLEDGED,
            OrderStatus.REJECTED: ExecutionEventKind.ORDER_REJECTED,
            OrderStatus.CANCEL_PENDING: ExecutionEventKind.ORDER_CANCEL_REQUESTED,
            OrderStatus.CANCELLED: ExecutionEventKind.ORDER_CANCELLED,
            OrderStatus.EXPIRED: ExecutionEventKind.ORDER_EXPIRED,
        }
        return mapping.get(status)

    def _audit(
        self,
        kind: ExecutionEventKind,
        message: str,
        *,
        order: BrokerOrder | None = None,
        fill: ExecutionFill | None = None,
    ) -> None:
        self._events.append(
            ExecutionEvent(
                sequence=self._sequence,
                timestamp=self._clock(),
                kind=kind,
                message=message,
                client_order_id=(
                    order.client_order_id
                    if order is not None
                    else fill.client_order_id if fill is not None else None
                ),
                broker_order_id=(
                    order.broker_order_id
                    if order is not None
                    else fill.broker_order_id if fill is not None else None
                ),
                execution_id=(fill.execution_id if fill is not None else None),
            )
        )
        self._sequence += 1
