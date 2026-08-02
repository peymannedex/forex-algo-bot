from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.execution.lifecycle import (
    InvalidOrderTransitionError,
    transition_allowed,
    validate_transition,
    with_status,
)
from fxbot.execution.models import BrokerOrder, OrderSide, OrderStatus, OrderType

BASE = datetime(2026, 1, 5, tzinfo=UTC)


def order(status: OrderStatus = OrderStatus.ACKNOWLEDGED) -> BrokerOrder:
    return BrokerOrder(
        "b-1",
        "c-1",
        "EURUSD",
        OrderSide.BUY,
        OrderType.LIMIT,
        status,
        2.0,
        0.0,
        None,
        BASE,
        BASE,
    )


def test_transition_table() -> None:
    assert transition_allowed(OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED)
    assert not transition_allowed(OrderStatus.FILLED, OrderStatus.CANCELLED)


def test_valid_partial_fill_transition() -> None:
    current = order()
    following = replace(
        current,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=1.0,
        average_fill_price=1.1,
        updated_at=BASE + timedelta(seconds=1),
    )
    validate_transition(current, following)


def test_fill_quantity_cannot_decrease() -> None:
    current = BrokerOrder(
        "b-1",
        "c-1",
        "EURUSD",
        OrderSide.BUY,
        OrderType.LIMIT,
        OrderStatus.PARTIALLY_FILLED,
        2,
        1,
        1.1,
        BASE,
        BASE,
    )
    following = replace(current, filled_quantity=0.5, updated_at=BASE + timedelta(seconds=1))
    with pytest.raises(InvalidOrderTransitionError, match="cannot decrease"):
        validate_transition(current, following)


def test_identity_cannot_change() -> None:
    current = order()
    following = replace(current, broker_order_id="different")
    with pytest.raises(InvalidOrderTransitionError, match="broker_order_id"):
        validate_transition(current, following)


def test_terminal_state_is_immutable() -> None:
    current = BrokerOrder(
        "b",
        "c",
        "EURUSD",
        OrderSide.BUY,
        OrderType.MARKET,
        OrderStatus.FILLED,
        1,
        1,
        1.1,
        BASE,
        BASE,
    )
    following = replace(current, updated_at=BASE + timedelta(seconds=1))
    with pytest.raises(InvalidOrderTransitionError, match="immutable"):
        validate_transition(current, following)


def test_with_status_builds_valid_state() -> None:
    current = order()
    following = with_status(
        current,
        OrderStatus.FILLED,
        updated_at=BASE + timedelta(seconds=1),
        filled_quantity=2,
        average_fill_price=1.2,
    )
    assert following.status is OrderStatus.FILLED
