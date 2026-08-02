from __future__ import annotations

from dataclasses import replace

import pytest

from fxbot.execution.idempotency import IdempotencyConflictError, InMemoryIdempotencyStore
from fxbot.execution.models import BrokerOrder, OrderStatus


def test_reserve_is_repeatable(market_intent) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryIdempotencyStore()
    first = store.reserve(market_intent)
    second = store.reserve(market_intent)
    assert first == second
    assert len(store) == 1


def test_key_conflict_is_rejected(market_intent) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryIdempotencyStore()
    store.reserve(market_intent)
    with pytest.raises(IdempotencyConflictError):
        store.reserve(replace(market_intent, quantity=2.0))


def test_bind_and_get(market_intent) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryIdempotencyStore()
    order = BrokerOrder(
        "b",
        market_intent.client_order_id,
        market_intent.symbol,
        market_intent.side,
        market_intent.order_type,
        OrderStatus.ACKNOWLEDGED,
        market_intent.quantity,
        0,
        None,
        market_intent.created_at,
        market_intent.created_at,
    )
    record = store.bind(market_intent, order)
    assert record.broker_order == order
    assert store.get("key-1") == record


def test_bind_conflict_is_rejected(market_intent) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryIdempotencyStore()
    order = BrokerOrder(
        "b",
        market_intent.client_order_id,
        market_intent.symbol,
        market_intent.side,
        market_intent.order_type,
        OrderStatus.ACKNOWLEDGED,
        1,
        0,
        None,
        market_intent.created_at,
        market_intent.created_at,
    )
    store.bind(market_intent, order)
    with pytest.raises(IdempotencyConflictError):
        store.bind(market_intent, replace(order, broker_order_id="other"))


def test_clear(market_intent) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryIdempotencyStore()
    store.reserve(market_intent)
    store.clear()
    assert len(store) == 0
