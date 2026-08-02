"""Explicit order-state transition rules for live execution."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from fxbot.execution.models import BrokerOrder, OrderStatus

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset({OrderStatus.VALIDATED, OrderStatus.REJECTED}),
    OrderStatus.VALIDATED: frozenset({OrderStatus.SUBMITTED, OrderStatus.REJECTED}),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.ACKNOWLEDGED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.CANCEL_PENDING: frozenset(
        {
            OrderStatus.CANCELLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.UNKNOWN: frozenset(
        {
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


class InvalidOrderTransitionError(ValueError):
    """Raised when a broker update violates the durable lifecycle."""


def transition_allowed(current: OrderStatus, following: OrderStatus) -> bool:
    """Return whether the lifecycle permits the requested transition."""

    current_status = OrderStatus(current)
    following_status = OrderStatus(following)
    return following_status in _ALLOWED_TRANSITIONS[current_status]


def validate_transition(current: BrokerOrder, following: BrokerOrder) -> None:
    """Validate identity, monotonic fill quantity, time, and status transition."""

    if current.broker_order_id != following.broker_order_id:
        raise InvalidOrderTransitionError("broker_order_id cannot change")
    if current.client_order_id != following.client_order_id:
        raise InvalidOrderTransitionError("client_order_id cannot change")
    if current.symbol != following.symbol or current.side is not following.side:
        raise InvalidOrderTransitionError("order instrument or side cannot change")
    if current.requested_quantity != following.requested_quantity:
        raise InvalidOrderTransitionError("requested_quantity cannot change")
    if following.updated_at < current.updated_at:
        raise InvalidOrderTransitionError("updated_at cannot move backwards")
    if following.filled_quantity + 1e-12 < current.filled_quantity:
        raise InvalidOrderTransitionError("filled_quantity cannot decrease")
    if current.status is following.status:
        if current.status.terminal and current != following:
            raise InvalidOrderTransitionError("terminal order states are immutable")
        return
    if not transition_allowed(current.status, following.status):
        raise InvalidOrderTransitionError(
            f"Invalid transition: {current.status.value} -> {following.status.value}"
        )


def with_status(
    order: BrokerOrder,
    status: OrderStatus,
    *,
    updated_at: datetime,
    filled_quantity: float | None = None,
    average_fill_price: float | None = None,
    rejection_reason: str | None = None,
) -> BrokerOrder:
    """Construct and validate a new immutable state from an existing order."""

    following = replace(
        order,
        status=OrderStatus(status),
        updated_at=updated_at,
        filled_quantity=(order.filled_quantity if filled_quantity is None else filled_quantity),
        average_fill_price=(
            order.average_fill_price if average_fill_price is None else average_fill_price
        ),
        rejection_reason=rejection_reason,
    )
    validate_transition(order, following)
    return following
