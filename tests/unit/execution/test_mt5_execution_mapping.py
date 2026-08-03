from datetime import UTC, datetime
from typing import Any

import pytest

from fxbot.execution.models import (
    OrderIntent,
    OrderSide,
    OrderType,
    Quote,
    TimeInForce,
)
from fxbot.execution.mt5_mapping import (
    MT5SymbolSpec,
    build_mt5_request,
    client_order_comment,
    normalize_price,
    normalize_volume,
    order_type_from_mt5,
    side_from_mt5_order_type,
    status_from_mt5_state,
    validate_entry_prices,
    validate_freeze_distance,
)


def intent(**kwargs: Any) -> OrderIntent:
    values: dict[str, Any] = {
        "client_order_id": "client-123",
        "idempotency_key": "idem",
        "symbol": "EURUSD",
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": 0.127,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(kwargs)
    return OrderIntent(**values)


def spec(**kwargs: Any) -> MT5SymbolSpec:
    values: dict[str, Any] = {
        "symbol": "EURUSD",
        "digits": 5,
        "point": 0.00001,
        "tick_size": 0.00001,
        "volume_min": 0.01,
        "volume_max": 10.0,
        "volume_step": 0.01,
        "stops_level_points": 10,
        "freeze_level_points": 5,
        "filling_mode": 2,
    }
    values.update(kwargs)
    return MT5SymbolSpec(**values)


def quote() -> Quote:
    return Quote(
        "EURUSD",
        1.1000,
        1.1002,
        datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_symbol_spec_from_mt5(client: Any) -> None:
    parsed = MT5SymbolSpec.from_mt5("eurusd", client.info)

    assert parsed.symbol == "EURUSD"
    assert parsed.stops_distance == pytest.approx(0.0001)


def test_volume_rounds_down_without_increasing_risk() -> None:
    assert normalize_volume(0.127, spec()) == pytest.approx(0.12)


def test_volume_clamps_to_maximum() -> None:
    assert normalize_volume(12.0, spec(volume_max=10.0)) == 10.0


def test_volume_below_minimum_rejected() -> None:
    with pytest.raises(ValueError, match="below broker minimum"):
        normalize_volume(0.001, spec())


def test_price_rounds_to_tick() -> None:
    assert normalize_price(1.100206, spec()) == pytest.approx(1.10021)


def test_comment_is_deterministic_and_bounded() -> None:
    first = client_order_comment("client order/123")

    assert first == client_order_comment("client order/123")
    assert len(first) <= 31
    assert first.startswith("fxb:")


def test_market_request_uses_ask_for_buy(client: Any) -> None:
    request = build_mt5_request(
        client,
        intent(),
        spec(),
        quote(),
        magic_number=51,
        deviation_points=10,
    )

    assert request["action"] == client.TRADE_ACTION_DEAL
    assert request["price"] == pytest.approx(1.1002)
    assert request["volume"] == pytest.approx(0.12)


def test_sell_market_uses_bid(client: Any) -> None:
    request = build_mt5_request(
        client,
        intent(side=OrderSide.SELL),
        spec(),
        quote(),
        magic_number=51,
        deviation_points=10,
    )

    assert request["price"] == pytest.approx(1.1)


def test_limit_request_mapping(client: Any) -> None:
    item = intent(order_type=OrderType.LIMIT, limit_price=1.0990)
    request = build_mt5_request(
        client,
        item,
        spec(),
        quote(),
        magic_number=51,
        deviation_points=10,
    )

    assert request["action"] == client.TRADE_ACTION_PENDING
    assert request["type"] == client.ORDER_TYPE_BUY_LIMIT


def test_stop_limit_mapping(client: Any) -> None:
    item = intent(
        order_type=OrderType.STOP_LIMIT,
        stop_price=1.1010,
        limit_price=1.1008,
    )
    request = build_mt5_request(
        client,
        item,
        spec(),
        quote(),
        magic_number=51,
        deviation_points=10,
    )

    assert request["type"] == client.ORDER_TYPE_BUY_STOP_LIMIT
    assert request["stoplimit"] == pytest.approx(1.1008)


def test_ioc_and_day_mapping(client: Any) -> None:
    ioc_request = build_mt5_request(
        client,
        intent(time_in_force=TimeInForce.IOC),
        spec(),
        quote(),
        magic_number=51,
        deviation_points=10,
    )
    day_request = build_mt5_request(
        client,
        intent(
            order_type=OrderType.LIMIT,
            limit_price=1.099,
            time_in_force=TimeInForce.DAY,
        ),
        spec(),
        quote(),
        magic_number=51,
        deviation_points=10,
    )

    assert ioc_request["type_filling"] == client.ORDER_FILLING_IOC
    assert day_request["type_time"] == client.ORDER_TIME_DAY


def test_pending_price_validation_rejects_too_close() -> None:
    with pytest.raises(ValueError, match="BUY_LIMIT"):
        validate_entry_prices(
            intent(order_type=OrderType.LIMIT, limit_price=1.10015),
            quote(),
            spec(),
        )


def test_freeze_distance_validation() -> None:
    with pytest.raises(ValueError, match="freeze"):
        validate_freeze_distance(1.10018, quote(), spec())


def test_reverse_mappings(client: Any) -> None:
    assert (
        side_from_mt5_order_type(client, client.ORDER_TYPE_SELL_STOP)
        is OrderSide.SELL
    )
    assert (
        order_type_from_mt5(client, client.ORDER_TYPE_BUY_LIMIT)
        is OrderType.LIMIT
    )
    assert (
        status_from_mt5_state(client, client.ORDER_STATE_PARTIAL).value
        == "partially_filled"
    )
