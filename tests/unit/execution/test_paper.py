from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.execution.broker import OrderNotFoundError, PermanentBrokerError
from fxbot.execution.models import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    TimeInForce,
)
from fxbot.execution.paper import PaperBroker, PaperBrokerConfig

BASE = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


def intent(
    order_type: OrderType,
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: float = 1.0,
    limit: float | None = None,
    stop: float | None = None,
    tif: TimeInForce = TimeInForce.GTC,
) -> OrderIntent:
    return OrderIntent(
        f"c-{order_type.value}-{side.value}-{quantity}",
        f"k-{order_type.value}-{side.value}-{quantity}",
        "EURUSD",
        side,
        order_type,
        quantity,
        BASE,
        time_in_force=tif,
        limit_price=limit,
        stop_price=stop,
    )


def test_market_buy_fills_at_ask_with_costs() -> None:
    broker = PaperBroker(PaperBrokerConfig(commission_per_unit=0.2, slippage=0.0001))
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(intent(OrderType.MARKET))
    fills = broker.drain_fills()
    assert order.status is OrderStatus.FILLED
    assert order.average_fill_price == pytest.approx(1.1003)
    assert fills[0].commission == pytest.approx(0.2)


def test_market_sell_fills_at_bid() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(intent(OrderType.MARKET, side=OrderSide.SELL))
    assert order.average_fill_price == pytest.approx(1.1)


def test_market_without_quote_is_rejected() -> None:
    broker = PaperBroker()
    with pytest.raises(PermanentBrokerError):
        broker.submit_order(intent(OrderType.MARKET))
    assert broker.orders[0].status is OrderStatus.REJECTED


def test_buy_limit_waits_then_fills() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(intent(OrderType.LIMIT, limit=1.0995))
    assert order.status is OrderStatus.ACKNOWLEDGED
    broker.update_quote(Quote("EURUSD", 1.0992, 1.0994, BASE + timedelta(seconds=1)))
    assert broker.get_order(order.broker_order_id).status is OrderStatus.FILLED


def test_sell_stop_triggers_on_bid() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(
        intent(OrderType.STOP, side=OrderSide.SELL, stop=1.0995)
    )
    broker.update_quote(Quote("EURUSD", 1.0994, 1.0996, BASE + timedelta(seconds=1)))
    assert broker.get_order(order.broker_order_id).status is OrderStatus.FILLED


def test_stop_limit_requires_both_conditions() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(
        intent(OrderType.STOP_LIMIT, stop=1.1010, limit=1.1008)
    )
    broker.update_quote(Quote("EURUSD", 1.1010, 1.1012, BASE + timedelta(seconds=1)))
    assert broker.get_order(order.broker_order_id).status is OrderStatus.ACKNOWLEDGED
    broker.update_quote(Quote("EURUSD", 1.1005, 1.1007, BASE + timedelta(seconds=2)))
    assert broker.get_order(order.broker_order_id).status is OrderStatus.FILLED


def test_partial_fills_across_quotes() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_fill_quantity_per_quote=1.0))
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(intent(OrderType.LIMIT, quantity=2, limit=1.1010))
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == 1.0
    broker.update_quote(Quote("EURUSD", 1.1, 1.1001, BASE + timedelta(seconds=1)))
    assert broker.get_order(order.broker_order_id).status is OrderStatus.FILLED
    assert len(broker.drain_fills()) == 2


def test_fok_cancels_when_full_quantity_unavailable() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_fill_quantity_per_quote=1.0))
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(
        intent(OrderType.LIMIT, quantity=2, limit=1.1010, tif=TimeInForce.FOK)
    )
    assert order.status is OrderStatus.CANCELLED
    assert broker.drain_fills() == ()


def test_ioc_cancels_unfilled_remainder() -> None:
    broker = PaperBroker(PaperBrokerConfig(max_fill_quantity_per_quote=1.0))
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(
        intent(OrderType.LIMIT, quantity=2, limit=1.1010, tif=TimeInForce.IOC)
    )
    assert order.status is OrderStatus.CANCELLED
    assert order.filled_quantity == 1.0


def test_day_order_expires_on_next_date() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(
        intent(OrderType.LIMIT, limit=1.0, tif=TimeInForce.DAY)
    )
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE + timedelta(days=1)))
    assert broker.get_order(order.broker_order_id).status is OrderStatus.EXPIRED


def test_cancel_and_open_order_listing() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    order = broker.submit_order(intent(OrderType.LIMIT, limit=1.0))
    assert broker.list_open_orders() == (order,)
    cancelled = broker.cancel_order(order.broker_order_id)
    assert cancelled.status is OrderStatus.CANCELLED
    assert broker.list_open_orders() == ()


def test_duplicate_client_order_returns_existing() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    original_intent = intent(OrderType.LIMIT, limit=1.0)
    first = broker.submit_order(original_intent)
    second = broker.submit_order(replace(original_intent, idempotency_key="new-key"))
    assert first == second
    assert len(broker.orders) == 1


def test_quote_ordering_and_unknown_order_errors() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    with pytest.raises(ValueError, match="chronological"):
        broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE - timedelta(seconds=1)))
    with pytest.raises(OrderNotFoundError):
        broker.get_order("missing")
