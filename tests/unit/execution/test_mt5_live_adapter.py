from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from fxbot.execution.adapters.mt5 import MT5BrokerAdapter, MT5ExecutionConfig
from fxbot.execution.broker import PermanentBrokerError, TransientBrokerError
from fxbot.execution.connection import MT5ConnectionManager
from fxbot.execution.models import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
)
from fxbot.execution.mt5_mapping import client_order_comment


def make_intent(**kwargs: Any) -> OrderIntent:
    values: dict[str, Any] = {
        "client_order_id": "c-1",
        "idempotency_key": "i-1",
        "symbol": "EURUSD",
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": 0.1,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(kwargs)
    return OrderIntent(**values)


def make_adapter(client: Any, **kwargs: Any) -> MT5BrokerAdapter:
    return MT5BrokerAdapter(
        MT5ConnectionManager(client=client),
        config=MT5ExecutionConfig(**kwargs),
    )


def raw_order(
    client: Any,
    *,
    ticket: int = 101,
    state: int | None = None,
    magic: int = 51001,
    comment: str = "x",
    volume_current: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticket=ticket,
        symbol="EURUSD",
        type=client.ORDER_TYPE_BUY_LIMIT,
        state=client.ORDER_STATE_PLACED if state is None else state,
        volume_initial=0.1,
        volume_current=volume_current,
        price_open=1.0990,
        time_setup=1704067200,
        time_done=1704067200,
        magic=magic,
        comment=comment,
    )


def raw_deal(
    client: Any,
    *,
    ticket: int = 501,
    order: int = 101,
    magic: int = 51001,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticket=ticket,
        order=order,
        symbol="EURUSD",
        type=client.DEAL_TYPE_BUY,
        volume=0.1,
        price=1.1002,
        time_msc=1704067200123,
        commission=-1.5,
        entry=0,
        magic=magic,
    )


def test_name(client: Any) -> None:
    assert make_adapter(client).name == "mt5"


def test_market_submission_returns_filled_order(client: Any) -> None:
    order = make_adapter(client).submit_order(make_intent())

    assert order.status is OrderStatus.FILLED
    assert order.broker_order_id == "101"
    assert order.filled_quantity == pytest.approx(0.1)


def test_submission_queues_fill(client: Any) -> None:
    adapter = make_adapter(client)
    adapter.submit_order(make_intent())

    fills = adapter.drain_fills()

    assert len(fills) == 1
    assert fills[0].execution_id == "501"


def test_pending_submission_acknowledged(client: Any) -> None:
    client.send_result = SimpleNamespace(
        retcode=client.TRADE_RETCODE_PLACED,
        order=102,
        deal=0,
        volume=0.0,
        price=0.0,
        comment="placed",
    )

    order = make_adapter(client).submit_order(
        make_intent(
            order_type=OrderType.LIMIT,
            limit_price=1.099,
        )
    )

    assert order.status is OrderStatus.ACKNOWLEDGED
    assert order.filled_quantity == 0.0


def test_dry_run_checks_but_does_not_send(client: Any) -> None:
    adapter = make_adapter(client, dry_run=True)

    order = adapter.submit_order(make_intent())

    assert order.broker_order_id.startswith("dry-")
    assert [item[0] for item in client.requests] == ["check"]


def test_invalid_broker_constraints_become_permanent(client: Any) -> None:
    with pytest.raises(PermanentBrokerError):
        make_adapter(client).submit_order(make_intent(quantity=0.001))


def test_transient_check_error(client: Any) -> None:
    client.check_result = SimpleNamespace(
        retcode=client.TRADE_RETCODE_PRICE_OFF,
        comment="off",
    )

    with pytest.raises(TransientBrokerError):
        make_adapter(client).submit_order(make_intent())


def test_cancel_dry_run(client: Any) -> None:
    adapter = make_adapter(client, dry_run=True)
    order = adapter.submit_order(make_intent())

    cancelled = adapter.cancel_order(order.broker_order_id)

    assert cancelled.status is OrderStatus.CANCELLED


def test_cancel_live_order(client: Any) -> None:
    client.send_result = SimpleNamespace(
        retcode=client.TRADE_RETCODE_PLACED,
        order=102,
        deal=0,
        volume=0.0,
        price=0.0,
        comment="placed",
    )
    adapter = make_adapter(client)
    order = adapter.submit_order(
        make_intent(
            order_type=OrderType.LIMIT,
            limit_price=1.099,
        )
    )
    client.send_result = SimpleNamespace(
        retcode=client.TRADE_RETCODE_DONE,
        order=102,
        deal=0,
        volume=0.0,
        price=0.0,
        comment="cancel",
    )

    cancelled = adapter.cancel_order(order.broker_order_id)

    assert cancelled.status is OrderStatus.CANCELLED


def test_list_open_orders_filters_magic(client: Any) -> None:
    client.open_orders = [
        raw_order(client, ticket=1, magic=51001),
        raw_order(client, ticket=2, magic=999),
    ]

    orders = make_adapter(client).list_open_orders()

    assert [order.broker_order_id for order in orders] == ["1"]


def test_find_order_by_client_id_from_history(client: Any) -> None:
    client.history_orders = [
        raw_order(
            client,
            ticket=77,
            comment=client_order_comment("client-x"),
        )
    ]

    order = make_adapter(client).find_order_by_client_id("client-x")

    assert order is not None
    assert order.client_order_id == "client-x"


def test_get_order_from_history(client: Any) -> None:
    client.history_orders = [
        raw_order(
            client,
            ticket=88,
            state=client.ORDER_STATE_CANCELED,
        )
    ]

    order = make_adapter(client).get_order("88")

    assert order.status is OrderStatus.CANCELLED


def test_history_fill_recovery(client: Any) -> None:
    client.deals = [raw_deal(client, ticket=901, order=88)]

    fills = make_adapter(client).recover_fills(
        datetime(2025, 1, 1, tzinfo=UTC)
    )

    assert fills[0].commission == pytest.approx(1.5)


def test_fill_drain_deduplicates_history(client: Any) -> None:
    client.deals = [raw_deal(client, ticket=501, order=101)]
    adapter = make_adapter(client)
    adapter.submit_order(make_intent())

    assert len(adapter.drain_fills()) == 1


def test_positions_are_signed(client: Any) -> None:
    client.positions = [
        SimpleNamespace(
            ticket=1,
            symbol="EURUSD",
            type=client.POSITION_TYPE_BUY,
            volume=0.2,
            price_open=1.1,
            profit=10.0,
            time=1704067200,
            magic=51001,
        ),
        SimpleNamespace(
            ticket=2,
            symbol="EURUSD",
            type=client.POSITION_TYPE_SELL,
            volume=0.1,
            price_open=1.2,
            profit=-5.0,
            time=1704067200,
            magic=51001,
        ),
    ]

    positions = make_adapter(client).snapshot_positions()

    assert [position.signed_quantity for position in positions] == [0.2, -0.1]


def test_update_quote_noop(client: Any) -> None:
    adapter = make_adapter(client)
    quote = Quote(
        "EURUSD",
        1.0,
        1.1,
        datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert adapter.update_quote(quote) == ()


def test_reduce_only_selects_opposite_position(client: Any) -> None:
    client.positions = [
        SimpleNamespace(
            ticket=42,
            symbol="EURUSD",
            type=client.POSITION_TYPE_SELL,
            volume=0.2,
            price_open=1.2,
            profit=0.0,
            time=1704067200,
            magic=51001,
        )
    ]
    adapter = make_adapter(client)

    adapter.submit_order(make_intent(reduce_only=True))

    send_request = next(
        request
        for kind, request in client.requests
        if kind == "send"
    )
    assert send_request["position"] == 42


def test_reduce_only_rejects_without_position(client: Any) -> None:
    with pytest.raises(PermanentBrokerError, match="reduce-only"):
        make_adapter(client).submit_order(make_intent(reduce_only=True))
