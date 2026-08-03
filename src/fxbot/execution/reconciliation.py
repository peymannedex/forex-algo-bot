"""Live MT5 order, fill, and position reconciliation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Protocol

from fxbot.execution.models import BrokerOrder, ExecutionFill


@dataclass(frozen=True, slots=True)
class MT5PositionSnapshot:
    """Broker position normalized for execution reconciliation."""

    position_id: str
    symbol: str
    signed_quantity: float
    average_price: float
    profit: float
    updated_at: datetime

    def __post_init__(self) -> None:
        position_id = self.position_id.strip()
        symbol = self.symbol.strip().upper()
        if not position_id or not symbol:
            raise ValueError("position_id and symbol cannot be empty")
        object.__setattr__(self, "position_id", position_id)
        object.__setattr__(self, "symbol", symbol)
        for name in ("signed_quantity", "average_price", "profit"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.average_price <= 0.0:
            raise ValueError("average_price must be positive")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(UTC))


class ReconciliationIssueKind(StrEnum):
    LOCAL_ACTIVE_ORDER_MISSING = "local_active_order_missing"
    BROKER_ORDER_UNKNOWN_LOCALLY = "broker_order_unknown_locally"
    POSITION_MISMATCH = "position_mismatch"
    MISSED_FILL_RECOVERED = "missed_fill_recovered"


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    kind: ReconciliationIssueKind
    message: str
    identifier: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ReconciliationIssueKind(self.kind))
        object.__setattr__(self, "message", self.message.strip() or self.kind.value)
        identifier = self.identifier.strip()
        if not identifier:
            raise ValueError("identifier cannot be empty")
        object.__setattr__(self, "identifier", identifier)


@dataclass(frozen=True, slots=True)
class MT5ReconciliationReport:
    checked_at: datetime
    broker_orders: tuple[BrokerOrder, ...]
    broker_positions: tuple[MT5PositionSnapshot, ...]
    recovered_fills: tuple[ExecutionFill, ...]
    issues: tuple[ReconciliationIssue, ...]

    def __post_init__(self) -> None:
        if self.checked_at.tzinfo is None or self.checked_at.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        object.__setattr__(self, "checked_at", self.checked_at.astimezone(UTC))

    @property
    def clean(self) -> bool:
        return not self.issues


class MT5ReconciliationSource(Protocol):
    def snapshot_open_orders(self) -> tuple[BrokerOrder, ...]: ...

    def snapshot_positions(self) -> tuple[MT5PositionSnapshot, ...]: ...

    def recover_fills(self, since: datetime) -> tuple[ExecutionFill, ...]: ...


class LiveMT5Reconciler:
    """Compare local execution state with authoritative MT5 state."""

    def __init__(self, source: MT5ReconciliationSource, *, quantity_tolerance: float = 1e-9) -> None:
        if quantity_tolerance < 0.0 or not isfinite(quantity_tolerance):
            raise ValueError("quantity_tolerance must be finite and non-negative")
        self.source = source
        self.quantity_tolerance = quantity_tolerance

    def reconcile(
        self,
        *,
        local_orders: tuple[BrokerOrder, ...],
        expected_positions: dict[str, float],
        since: datetime,
        checked_at: datetime | None = None,
    ) -> MT5ReconciliationReport:
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since must be timezone-aware")
        now = (checked_at or datetime.now(UTC)).astimezone(UTC)
        broker_orders = self.source.snapshot_open_orders()
        broker_positions = self.source.snapshot_positions()
        recovered_fills = self.source.recover_fills(since.astimezone(UTC))

        issues: list[ReconciliationIssue] = []
        broker_by_id = {order.broker_order_id: order for order in broker_orders}
        local_by_id = {order.broker_order_id: order for order in local_orders}

        for order in local_orders:
            if order.status.active and order.broker_order_id not in broker_by_id:
                issues.append(
                    ReconciliationIssue(
                        ReconciliationIssueKind.LOCAL_ACTIVE_ORDER_MISSING,
                        "Locally active order is absent from MT5 open orders",
                        order.broker_order_id,
                    )
                )
        for order in broker_orders:
            if order.broker_order_id not in local_by_id:
                issues.append(
                    ReconciliationIssue(
                        ReconciliationIssueKind.BROKER_ORDER_UNKNOWN_LOCALLY,
                        "MT5 open order is not present in local execution state",
                        order.broker_order_id,
                    )
                )

        actual_by_symbol: dict[str, float] = {}
        for position in broker_positions:
            actual_by_symbol[position.symbol] = (
                actual_by_symbol.get(position.symbol, 0.0) + position.signed_quantity
            )
        all_symbols = set(actual_by_symbol) | {symbol.strip().upper() for symbol in expected_positions}
        for symbol in sorted(all_symbols):
            expected = float(expected_positions.get(symbol, expected_positions.get(symbol.lower(), 0.0)))
            actual = actual_by_symbol.get(symbol, 0.0)
            if abs(expected - actual) > self.quantity_tolerance:
                issues.append(
                    ReconciliationIssue(
                        ReconciliationIssueKind.POSITION_MISMATCH,
                        f"Expected signed quantity {expected}, MT5 reports {actual}",
                        symbol,
                    )
                )

        for fill in recovered_fills:
            issues.append(
                ReconciliationIssue(
                    ReconciliationIssueKind.MISSED_FILL_RECOVERED,
                    "Historical MT5 deal recovered during reconciliation",
                    fill.execution_id,
                )
            )

        issues.sort(key=lambda item: (item.kind.value, item.identifier))
        return MT5ReconciliationReport(
            checked_at=now,
            broker_orders=broker_orders,
            broker_positions=broker_positions,
            recovered_fills=recovered_fills,
            issues=tuple(issues),
        )


__all__ = [
    "LiveMT5Reconciler",
    "MT5PositionSnapshot",
    "MT5ReconciliationReport",
    "MT5ReconciliationSource",
    "ReconciliationIssue",
    "ReconciliationIssueKind",
]
