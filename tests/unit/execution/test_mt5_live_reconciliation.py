from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

import pytest

from fxbot.execution.models import (
    BrokerOrder,
    ExecutionFill,
    OrderSide,
    OrderStatus,
    OrderType,
)
from fxbot.execution.reconciliation import (
    LiveMT5Reconciler,
    MT5PositionSnapshot,
    ReconciliationIssueKind,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def order(
    identifier: str,
    status: OrderStatus = OrderStatus.ACKNOWLEDGED,
) -> BrokerOrder:
    return BrokerOrder(
        broker_order_id=identifier,
        client_order_id=f"client-{identifier}",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=status,
        requested_quantity=0.1,
        filled_quantity=0.0,
        average_fill_price=None,
        submitted_at=NOW,
        updated_at=NOW,
    )


def fill(identifier: str = "f1") -> ExecutionFill:
    return ExecutionFill(
        execution_id=identifier,
        broker_order_id="1",
        client_order_id="client-1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=0.1,
        price=1.1,
        executed_at=NOW,
    )


class ReconciliationSource(Protocol):
    def snapshot_open_orders(self) -> tuple[BrokerOrder, ...]: ...

    def snapshot_positions(self) -> tuple[MT5PositionSnapshot, ...]: ...

    def recover_fills(self, since: datetime) -> tuple[ExecutionFill, ...]: ...


class Source:
    def __init__(
        self,
        orders: tuple[BrokerOrder, ...] = (),
        positions: tuple[MT5PositionSnapshot, ...] = (),
        fills: tuple[ExecutionFill, ...] = (),
    ) -> None:
        self.orders = orders
        self.positions = positions
        self.fills = fills

    def snapshot_open_orders(self) -> tuple[BrokerOrder, ...]:
        return self.orders

    def snapshot_positions(self) -> tuple[MT5PositionSnapshot, ...]:
        return self.positions

    def recover_fills(self, since: datetime) -> tuple[ExecutionFill, ...]:
        return self.fills


def test_clean_reconciliation() -> None:
    local_orders = (order("1"),)
    positions = (
        MT5PositionSnapshot(
            position_id="p",
            symbol="EURUSD",
            signed_quantity=0.1,
            average_price=1.1,
            profit=0.0,
            updated_at=NOW,
        ),
    )

    report = LiveMT5Reconciler(Source(local_orders, positions)).reconcile(
        local_orders=local_orders,
        expected_positions={"EURUSD": 0.1},
        since=NOW,
    )

    assert report.clean


def test_missing_local_order_issue() -> None:
    report = LiveMT5Reconciler(Source()).reconcile(
        local_orders=(order("1"),),
        expected_positions={},
        since=NOW,
    )

    assert (
        report.issues[0].kind
        is ReconciliationIssueKind.LOCAL_ACTIVE_ORDER_MISSING
    )


def test_unknown_broker_order_issue() -> None:
    report = LiveMT5Reconciler(Source((order("2"),))).reconcile(
        local_orders=(),
        expected_positions={},
        since=NOW,
    )

    assert (
        report.issues[0].kind
        is ReconciliationIssueKind.BROKER_ORDER_UNKNOWN_LOCALLY
    )


def test_position_mismatch_issue() -> None:
    positions = (
        MT5PositionSnapshot(
            position_id="p",
            symbol="EURUSD",
            signed_quantity=0.2,
            average_price=1.1,
            profit=0.0,
            updated_at=NOW,
        ),
    )

    report = LiveMT5Reconciler(Source(positions=positions)).reconcile(
        local_orders=(),
        expected_positions={"EURUSD": 0.1},
        since=NOW,
    )

    assert report.issues[0].kind is ReconciliationIssueKind.POSITION_MISMATCH


def test_recovered_fill_is_reported() -> None:
    report = LiveMT5Reconciler(Source(fills=(fill(),))).reconcile(
        local_orders=(),
        expected_positions={},
        since=NOW,
    )

    assert report.recovered_fills
    assert (
        report.issues[0].kind
        is ReconciliationIssueKind.MISSED_FILL_RECOVERED
    )


def test_naive_since_rejected() -> None:
    with pytest.raises(ValueError):
        LiveMT5Reconciler(Source()).reconcile(
            local_orders=(),
            expected_positions={},
            since=datetime(2026, 1, 1),
        )
