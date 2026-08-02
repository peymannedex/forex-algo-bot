from datetime import UTC, datetime

import pytest

from fxbot.backtest.events import (
    MarketEvent,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    SimulatedFill,
    TimeInForce,
    market_record_time,
)
from fxbot.domain.models import Tick

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def tick() -> Tick:
    return Tick(symbol="eurusd", event_time=BASE, bid=1.1, ask=1.1002)


def test_market_event_uses_record_availability_time() -> None:
    record = tick()
    event = MarketEvent(sequence=0, timestamp=record.event_time, record=record)
    assert event.symbol == "EURUSD"
    assert market_record_time(record) == BASE


def test_market_event_rejects_mismatched_timestamp() -> None:
    with pytest.raises(ValueError, match="availability time"):
        MarketEvent(
            sequence=0,
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            record=tick(),
        )


def test_order_request_validates_type_specific_prices() -> None:
    with pytest.raises(ValueError, match="LIMIT orders require"):
        OrderRequest(
            order_id="o1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=1.0,
            submitted_at=BASE,
        )
    with pytest.raises(ValueError, match="MARKET orders cannot"):
        OrderRequest(
            order_id="o2",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=1.0,
            submitted_at=BASE,
            limit_price=1.0,
        )


def test_order_request_normalizes_metadata_and_symbol() -> None:
    request = OrderRequest(
        order_id=" order ",
        symbol="eurusd",
        side="buy",
        order_type="stop",
        volume=0.5,
        submitted_at=BASE,
        stop_price=1.2,
        time_in_force=TimeInForce.IOC,
        metadata=(("z", "2"), ("a", "1")),
    )
    assert request.order_id == "order"
    assert request.symbol == "EURUSD"
    assert request.metadata == (("a", "1"), ("z", "2"))


def test_order_status_terminal_property() -> None:
    assert OrderStatus.FILLED.terminal
    assert OrderStatus.REJECTED.terminal
    assert not OrderStatus.PENDING.terminal


def test_fill_validates_non_negative_costs() -> None:
    with pytest.raises(ValueError, match="commission"):
        SimulatedFill(
            fill_id="f1",
            order_id="o1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            volume=1.0,
            price=1.1,
            timestamp=BASE,
            sequence=1,
            commission=-1.0,
        )
