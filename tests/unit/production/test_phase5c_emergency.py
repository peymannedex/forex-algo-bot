from datetime import UTC, datetime

from fxbot.execution.models import (
    BrokerOrder,
    OrderSide,
    OrderStatus,
    OrderType,
)
from fxbot.execution.reconciliation import MT5PositionSnapshot
from fxbot.execution.safety import ExecutionControl
from fxbot.production.emergency import EmergencyController

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def broker_order(
    broker_order_id: str,
    client_order_id: str,
) -> BrokerOrder:
    return BrokerOrder(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.ACKNOWLEDGED,
        requested_quantity=0.1,
        filled_quantity=0.0,
        average_fill_price=None,
        submitted_at=NOW,
        updated_at=NOW,
    )


class Broker:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.intents = []

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        return (
            broker_order("1", "c1"),
            broker_order("2", "c2"),
        )

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        self.cancelled.append(broker_order_id)
        return next(
            item
            for item in self.list_open_orders()
            if item.broker_order_id == broker_order_id
        )

    def submit_order(self, intent):
        self.intents.append(intent)
        return broker_order("x", intent.client_order_id)

    def snapshot_positions(self) -> tuple[MT5PositionSnapshot, ...]:
        return (
            MT5PositionSnapshot(
                "p1",
                "EURUSD",
                0.2,
                1.1,
                0.0,
                NOW,
            ),
            MT5PositionSnapshot(
                "p2",
                "GBPUSD",
                -0.1,
                1.2,
                0.0,
                NOW,
            ),
        )


def test_emergency_cancels_and_flattens() -> None:
    control = ExecutionControl.armed()
    broker = Broker()
    controller = EmergencyController(
        control=control,
        broker=broker,
    )

    result = controller.trigger(
        "operator",
        flatten_positions=True,
        now=NOW,
    )

    assert not control.state.enabled
    assert result.cancelled_orders == 2
    assert result.flatten_orders == 2
    assert broker.intents[0].reduce_only
    assert {intent.side.value for intent in broker.intents} == {"buy", "sell"}
