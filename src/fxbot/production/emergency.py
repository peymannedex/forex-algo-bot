"""Emergency cancel, flatten, and stop workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from fxbot.execution.models import (
    BrokerOrder,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from fxbot.execution.reconciliation import MT5PositionSnapshot
from fxbot.execution.safety import ExecutionControl


class EmergencyBroker(Protocol):
    """Broker capabilities required for an emergency response."""

    def list_open_orders(self) -> tuple[BrokerOrder, ...]: ...

    def cancel_order(self, broker_order_id: str) -> BrokerOrder: ...

    def snapshot_positions(self) -> tuple[MT5PositionSnapshot, ...]: ...

    def submit_order(self, intent: OrderIntent) -> BrokerOrder: ...


@dataclass(frozen=True, slots=True)
class EmergencyResult:
    reason: str
    cancelled_orders: int
    flatten_orders: int
    errors: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.errors


class EmergencyController:
    """Trip normal routing, then cancel and optionally flatten directly at the broker."""

    def __init__(
        self,
        *,
        control: ExecutionControl,
        broker: EmergencyBroker,
    ) -> None:
        self.control = control
        self.broker = broker

    def trigger(
        self,
        reason: str,
        *,
        cancel_orders: bool = True,
        flatten_positions: bool = False,
        now: datetime | None = None,
    ) -> EmergencyResult:
        normalized_reason = reason.strip() or "emergency stop"
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)

        # Disable ordinary strategy routing before touching broker state. Emergency
        # flatten orders intentionally go directly through the broker adapter so
        # they cannot be blocked by the shared kill switch.
        self.control.trip(normalized_reason)

        cancelled = 0
        flattened = 0
        errors: list[str] = []

        if cancel_orders:
            for order in self.broker.list_open_orders():
                try:
                    self.broker.cancel_order(order.broker_order_id)
                    cancelled += 1
                except Exception as exc:
                    errors.append(
                        f"cancel {order.broker_order_id}: {type(exc).__name__}: {exc}"
                    )

        if flatten_positions:
            for position in self.broker.snapshot_positions():
                if abs(position.signed_quantity) <= 1e-12:
                    continue
                side = (
                    OrderSide.SELL
                    if position.signed_quantity > 0.0
                    else OrderSide.BUY
                )
                suffix = int(timestamp.timestamp() * 1_000_000)
                client_id = f"emergency-{position.position_id}-{suffix}"
                intent = OrderIntent(
                    client_order_id=client_id,
                    idempotency_key=client_id,
                    symbol=position.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=abs(position.signed_quantity),
                    created_at=timestamp,
                    time_in_force=TimeInForce.IOC,
                    reduce_only=True,
                    strategy_id="emergency",
                    metadata=(("reason", normalized_reason),),
                )
                try:
                    self.broker.submit_order(intent)
                    flattened += 1
                except Exception as exc:
                    errors.append(
                        f"flatten {position.position_id}: {type(exc).__name__}: {exc}"
                    )

        return EmergencyResult(
            reason=normalized_reason,
            cancelled_orders=cancelled,
            flatten_orders=flattened,
            errors=tuple(errors),
        )
