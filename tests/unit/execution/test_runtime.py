from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.execution.broker import ExecutionError, ensure_unique_fills
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionFill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
)
from fxbot.execution.paper import PaperBroker
from fxbot.execution.runtime import ExecutionRuntime

BASE = datetime(2026, 1, 5, tzinfo=UTC)


class FillCollector:
    def __init__(self) -> None:
        self.items: list[ExecutionFill] = []

    def on_fill(self, fill: ExecutionFill) -> None:
        self.items.append(fill)


class OrderCollector:
    def __init__(self) -> None:
        self.items: list[BrokerOrder] = []

    def on_order(self, order: BrokerOrder) -> None:
        self.items.append(order)


def test_sync_dispatches_fill_and_order() -> None:
    broker = PaperBroker()
    collector = FillCollector()
    orders = OrderCollector()
    runtime = ExecutionRuntime(
        broker,
        fill_sinks=(collector,),
        order_sinks=(orders,),
        clock=lambda: BASE,
    )
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(
        OrderIntent(
            "c",
            "k",
            "EURUSD",
            OrderSide.BUY,
            OrderType.MARKET,
            1,
            BASE,
        )
    )
    runtime.observe_order(order)
    result = runtime.sync()
    assert result.new_fills == 1
    assert collector.items[0].client_order_id == "c"
    assert runtime.known_orders[0].status is OrderStatus.FILLED


def test_exact_order_duplicate_is_ignored() -> None:
    broker = PaperBroker()
    runtime = ExecutionRuntime(broker)
    order = BrokerOrder(
        "b",
        "c",
        "EURUSD",
        OrderSide.BUY,
        OrderType.LIMIT,
        OrderStatus.ACKNOWLEDGED,
        1,
        0,
        None,
        BASE,
        BASE,
    )
    assert runtime.observe_order(order)
    assert not runtime.observe_order(order)


def test_invalid_order_regression_raises() -> None:
    broker = PaperBroker()
    runtime = ExecutionRuntime(broker)
    current = BrokerOrder(
        "b",
        "c",
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
    runtime.observe_order(current)
    regressed = replace(current, filled_quantity=0.5, updated_at=BASE + timedelta(seconds=1))
    with pytest.raises(ValueError):
        runtime.observe_order(regressed)


def test_ensure_unique_fills_rejects_duplicates() -> None:
    fill = ExecutionFill(
        "e",
        "b",
        "c",
        "EURUSD",
        OrderSide.BUY,
        1,
        1.1,
        BASE,
    )
    with pytest.raises(ExecutionError, match="Duplicate"):
        ensure_unique_fills((fill, fill))


def test_sync_without_activity_is_empty() -> None:
    runtime = ExecutionRuntime(PaperBroker())
    result = runtime.sync()
    assert result.new_fills == 0
    assert result.order_updates == 0
    assert result.warnings == 0
