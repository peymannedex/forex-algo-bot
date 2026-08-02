from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.models import SymbolSpec
from fxbot.risk.models import BrokerVolumeConstraints, InstrumentRiskSpec, TradeSide
from fxbot.risk.position_sizing import IdentityCurrencyConverter
from fxbot.risk.positions import PositionLifecycle, PositionStatus
from fxbot.risk.reconciliation import (
    BrokerFill,
    BrokerPositionSnapshot,
    FillAction,
    FillReconciler,
    PositionLedger,
    PositionReconciler,
    ReconciliationIssueCode,
    ReconciliationStatus,
)

NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


def instrument(symbol: str = "EURUSD") -> InstrumentRiskSpec:
    base, quote = symbol[:3], symbol[3:]
    return InstrumentRiskSpec(
        symbol=SymbolSpec(symbol, base, quote, 5, 0.00001, 0.0001),
        volume=BrokerVolumeConstraints(0.01, 100, 0.01),
    )


def fill(
    *,
    fill_id: str = "f1",
    action: FillAction = FillAction.ENTRY,
    position_id: str = "p1",
    side: TradeSide = TradeSide.LONG,
    volume: float = 1.0,
    price: float = 1.10,
    symbol: str = "EURUSD",
    at: datetime = NOW,
) -> BrokerFill:
    return BrokerFill(
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        position_id=position_id,
        instrument=instrument(symbol),
        position_side=side,
        action=action,
        volume=volume,
        price=price,
        filled_at=at,
        stop_price=1.095 if action is FillAction.ENTRY and side is TradeSide.LONG else None,
    )


def reconciler() -> FillReconciler:
    return FillReconciler(
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )


def test_entry_fill_creates_position_and_records_fill_id() -> None:
    result = reconciler().apply(PositionLedger(), fill())
    assert result.status is ReconciliationStatus.APPLIED
    assert result.applied
    assert result.position is not None
    assert result.position.status is PositionStatus.OPEN
    assert result.ledger.processed_fill_ids == ("f1",)
    assert result.ledger.get("p1") == result.position


def test_duplicate_fill_is_idempotent() -> None:
    first = reconciler().apply(PositionLedger(), fill())
    second = reconciler().apply(first.ledger, fill())
    assert second.status is ReconciliationStatus.DUPLICATE
    assert second.ledger is first.ledger
    assert second.issues[0].code is ReconciliationIssueCode.DUPLICATE_FILL


def test_second_entry_fill_averages_existing_position() -> None:
    first = reconciler().apply(PositionLedger(), fill(volume=1, price=1.10))
    second = reconciler().apply(
        first.ledger,
        fill(fill_id="f2", volume=1, price=1.12, at=NOW + timedelta(minutes=1)),
    )
    assert second.position is not None
    assert second.position.open_volume == 2
    assert second.position.average_entry_price == pytest.approx(1.11)


def test_partial_and_final_exit_fills() -> None:
    opened = reconciler().apply(PositionLedger(), fill(volume=2))
    partial = reconciler().apply(
        opened.ledger,
        fill(
            fill_id="f2",
            action=FillAction.EXIT,
            volume=0.5,
            price=1.11,
            at=NOW + timedelta(minutes=1),
        ),
    )
    assert partial.position is not None
    assert partial.position.status is PositionStatus.PARTIALLY_CLOSED
    final = reconciler().apply(
        partial.ledger,
        fill(
            fill_id="f3",
            action=FillAction.EXIT,
            volume=1.5,
            price=1.12,
            at=NOW + timedelta(minutes=2),
        ),
    )
    assert final.position is not None
    assert final.position.status is PositionStatus.CLOSED
    assert final.position.realized_pnl_account == pytest.approx(3_500)


def test_unknown_exit_is_rejected_without_marking_fill_processed() -> None:
    result = reconciler().apply(
        PositionLedger(),
        fill(action=FillAction.EXIT, price=1.11),
    )
    assert result.status is ReconciliationStatus.REJECTED
    assert result.issues[0].code is ReconciliationIssueCode.UNKNOWN_POSITION
    assert result.ledger.processed_fill_ids == ()


def test_symbol_and_side_mismatch_are_rejected() -> None:
    first = reconciler().apply(PositionLedger(), fill())
    wrong_symbol = reconciler().apply(
        first.ledger,
        fill(fill_id="f2", symbol="GBPUSD", at=NOW + timedelta(minutes=1)),
    )
    wrong_side = reconciler().apply(
        first.ledger,
        fill(fill_id="f3", side=TradeSide.SHORT, at=NOW + timedelta(minutes=1)),
    )
    assert wrong_symbol.issues[0].code is ReconciliationIssueCode.SYMBOL_MISMATCH
    assert wrong_side.issues[0].code is ReconciliationIssueCode.SIDE_MISMATCH


def test_over_close_is_rejected() -> None:
    first = reconciler().apply(PositionLedger(), fill(volume=1))
    result = reconciler().apply(
        first.ledger,
        fill(
            fill_id="f2",
            action=FillAction.EXIT,
            volume=2,
            price=1.11,
            at=NOW + timedelta(minutes=1),
        ),
    )
    assert result.status is ReconciliationStatus.REJECTED
    assert result.issues[0].code is ReconciliationIssueCode.OVER_CLOSE


def test_stale_fill_becomes_invalid_transition() -> None:
    first = reconciler().apply(PositionLedger(), fill(at=NOW))
    stale = reconciler().apply(
        first.ledger,
        fill(fill_id="f2", at=NOW - timedelta(seconds=1)),
    )
    assert stale.issues[0].code is ReconciliationIssueCode.INVALID_TRANSITION


def test_ledger_rejects_duplicate_position_and_fill_ids() -> None:
    position = PositionLifecycle.open(
        position_id="p",
        instrument=instrument(),
        side=TradeSide.LONG,
        volume=1,
        entry_price=1.10,
        opened_at=NOW,
        stop_price=1.095,
    )
    with pytest.raises(ValueError, match="Position IDs"):
        PositionLedger(positions=(position, position))
    with pytest.raises(ValueError, match="fill IDs"):
        PositionLedger(processed_fill_ids=("f", "f"))


def test_broker_fill_validates_timestamp_and_values() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        fill(at=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        fill(volume=0)


def internal_position(position_id: str = "p1"):
    return PositionLifecycle.open(
        position_id=position_id,
        instrument=instrument(),
        side=TradeSide.LONG,
        volume=1,
        entry_price=1.10,
        opened_at=NOW,
        stop_price=1.095,
    )


def broker_position(
    position_id: str = "p1",
    *,
    symbol: str = "EURUSD",
    side: TradeSide = TradeSide.LONG,
    volume: float = 1,
    entry: float = 1.10,
    stop: float | None = 1.095,
) -> BrokerPositionSnapshot:
    return BrokerPositionSnapshot(
        position_id=position_id,
        symbol=symbol,
        side=side,
        volume=volume,
        average_entry_price=entry,
        stop_price=stop,
    )


def test_position_audit_returns_no_issues_for_matching_snapshots() -> None:
    assert PositionReconciler().audit(
        (internal_position(),),
        (broker_position(),),
    ) == ()


def test_position_audit_reports_missing_and_extra_positions() -> None:
    issues = PositionReconciler().audit(
        (internal_position("internal"),),
        (broker_position("broker"),),
    )
    assert {item.code for item in issues} == {
        ReconciliationIssueCode.EXTRA_INTERNAL_POSITION,
        ReconciliationIssueCode.MISSING_INTERNAL_POSITION,
    }


def test_position_audit_reports_all_field_mismatches() -> None:
    issues = PositionReconciler().audit(
        (internal_position(),),
        (
            broker_position(
                symbol="GBPUSD",
                side=TradeSide.SHORT,
                volume=2,
                entry=1.20,
                stop=1.21,
            ),
        ),
    )
    codes = {item.code for item in issues}
    assert ReconciliationIssueCode.SYMBOL_MISMATCH in codes
    assert ReconciliationIssueCode.SIDE_MISMATCH in codes
    assert ReconciliationIssueCode.VOLUME_MISMATCH in codes
    assert ReconciliationIssueCode.ENTRY_PRICE_MISMATCH in codes
    assert ReconciliationIssueCode.STOP_PRICE_MISMATCH in codes


def test_position_audit_respects_tolerances() -> None:
    issues = PositionReconciler().audit(
        (internal_position(),),
        (broker_position(volume=1.0000001, entry=1.1000001, stop=1.0950001),),
        volume_tolerance=1e-6,
        price_tolerance=1e-6,
    )
    assert issues == ()
    with pytest.raises(ValueError, match="non-negative"):
        PositionReconciler().audit((), (), volume_tolerance=-1)
