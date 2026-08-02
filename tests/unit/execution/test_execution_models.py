from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxbot.execution.models import (
    BrokerOrder,
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionFill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    RiskDecision,
    TimeInForce,
)

BASE = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


def test_market_intent_normalizes_and_fingerprints() -> None:
    intent = OrderIntent(
        " id-1 ",
        " key ",
        " eurusd ",
        OrderSide.BUY,
        OrderType.MARKET,
        2,
        BASE,
        metadata=(("z", "1"), ("a", "2")),
    )
    assert intent.symbol == "EURUSD"
    assert intent.metadata == (("a", "2"), ("z", "1"))
    assert len(intent.semantic_fingerprint) == 64


@pytest.mark.parametrize(
    ("order_type", "limit_price", "stop_price"),
    [
        (OrderType.MARKET, 1.0, None),
        (OrderType.LIMIT, None, None),
        (OrderType.STOP, None, None),
        (OrderType.STOP_LIMIT, 1.0, None),
    ],
)
def test_invalid_price_contracts(
    order_type: OrderType,
    limit_price: float | None,
    stop_price: float | None,
) -> None:
    with pytest.raises(ValueError):
        OrderIntent(
            "id",
            "key",
            "EURUSD",
            OrderSide.BUY,
            order_type,
            1,
            BASE,
            limit_price=limit_price,
            stop_price=stop_price,
        )


def test_duplicate_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate metadata"):
        OrderIntent(
            "id",
            "key",
            "EURUSD",
            OrderSide.BUY,
            OrderType.MARKET,
            1,
            BASE,
            metadata=(("x", "1"), ("x", "2")),
        )


def test_broker_order_computes_remaining_quantity() -> None:
    order = BrokerOrder(
        "broker",
        "client",
        "EURUSD",
        OrderSide.BUY,
        OrderType.LIMIT,
        OrderStatus.PARTIALLY_FILLED,
        2,
        0.5,
        1.1,
        BASE,
        BASE,
    )
    assert order.remaining_quantity == pytest.approx(1.5)


def test_broker_order_rejects_inconsistent_filled_state() -> None:
    with pytest.raises(ValueError, match="full requested quantity"):
        BrokerOrder(
            "broker",
            "client",
            "EURUSD",
            OrderSide.BUY,
            OrderType.MARKET,
            OrderStatus.FILLED,
            2,
            1,
            1.1,
            BASE,
            BASE,
        )


def test_execution_fill_validates_positive_values() -> None:
    fill = ExecutionFill(
        "exec",
        "broker",
        "client",
        "eurusd",
        OrderSide.SELL,
        1,
        1.2,
        BASE,
        commission=0.1,
    )
    assert fill.symbol == "EURUSD"
    with pytest.raises(ValueError):
        ExecutionFill("e", "b", "c", "x", OrderSide.BUY, 0, 1, BASE)


def test_quote_metrics_and_ordering() -> None:
    quote = Quote("eurusd", 1.1, 1.2, BASE)
    assert quote.spread == pytest.approx(0.1)
    assert quote.mid == pytest.approx(1.15)
    with pytest.raises(ValueError, match="ask cannot"):
        Quote("EURUSD", 1.2, 1.1, BASE)


def test_timezone_awareness_is_required() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Quote("EURUSD", 1.1, 1.2, datetime(2026, 1, 1))


def test_risk_decision_contract() -> None:
    assert RiskDecision(True, "ok", 0.5).approved_quantity == 0.5
    with pytest.raises(ValueError):
        RiskDecision(False, "no", 1.0)


def test_execution_event_normalizes() -> None:
    event = ExecutionEvent(
        0,
        BASE,
        ExecutionEventKind.FILL,
        "",
        client_order_id=" c ",
    )
    assert event.message == "fill"
    assert event.client_order_id == "c"


def test_enums_expose_expected_properties() -> None:
    assert OrderSide.BUY.opposite is OrderSide.SELL
    assert OrderSide.SELL.sign == -1.0
    assert OrderStatus.FILLED.terminal
    assert OrderStatus.ACKNOWLEDGED.active
    assert TimeInForce("gtc") is TimeInForce.GTC
