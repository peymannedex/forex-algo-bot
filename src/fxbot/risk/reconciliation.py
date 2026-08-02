"""Idempotent broker-fill application and position-state reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isclose, isfinite

from fxbot.risk.models import InstrumentRiskSpec, TradeSide
from fxbot.risk.position_sizing import CurrencyConverter
from fxbot.risk.positions import (
    ManagedPosition,
    PositionLifecycle,
    PositionLifecycleError,
)


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _finite(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class FillAction(StrEnum):
    """Whether a fill adds to or reduces a position."""

    ENTRY = "entry"
    EXIT = "exit"


class ReconciliationStatus(StrEnum):
    """Outcome of processing a broker fill."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class ReconciliationIssueCode(StrEnum):
    """Machine-readable mismatch or invalid-transition classification."""

    DUPLICATE_FILL = "duplicate_fill"
    UNKNOWN_POSITION = "unknown_position"
    SYMBOL_MISMATCH = "symbol_mismatch"
    SIDE_MISMATCH = "side_mismatch"
    OVER_CLOSE = "over_close"
    INVALID_TRANSITION = "invalid_transition"
    MISSING_INTERNAL_POSITION = "missing_internal_position"
    EXTRA_INTERNAL_POSITION = "extra_internal_position"
    VOLUME_MISMATCH = "volume_mismatch"
    ENTRY_PRICE_MISMATCH = "entry_price_mismatch"
    STOP_PRICE_MISMATCH = "stop_price_mismatch"


@dataclass(frozen=True, slots=True)
class BrokerFill:
    """Normalized broker execution event used for lifecycle reconciliation."""

    fill_id: str
    order_id: str
    position_id: str
    instrument: InstrumentRiskSpec
    position_side: TradeSide
    action: FillAction
    volume: float
    price: float
    filled_at: datetime
    commission_account: float = 0.0
    swap_account: float = 0.0
    stop_price: float | None = None
    take_profit_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_id", _identifier(self.fill_id, "fill_id"))
        object.__setattr__(self, "order_id", _identifier(self.order_id, "order_id"))
        object.__setattr__(self, "position_id", _identifier(self.position_id, "position_id"))
        object.__setattr__(self, "position_side", TradeSide(self.position_side))
        object.__setattr__(self, "action", FillAction(self.action))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(self, "price", _positive(self.price, "price"))
        object.__setattr__(self, "filled_at", _utc(self.filled_at, "filled_at"))
        object.__setattr__(
            self,
            "commission_account",
            _non_negative(self.commission_account, "commission_account"),
        )
        object.__setattr__(self, "swap_account", _finite(self.swap_account, "swap_account"))
        for name in ("stop_price", "take_profit_price"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    """Broker-reported open position used for state audits."""

    position_id: str
    symbol: str
    side: TradeSide
    volume: float
    average_entry_price: float
    stop_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_id", _identifier(self.position_id, "position_id"))
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", TradeSide(self.side))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(
            self,
            "average_entry_price",
            _positive(self.average_entry_price, "average_entry_price"),
        )
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _positive(self.stop_price, "stop_price"))


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    """One machine-readable fill or broker-state discrepancy."""

    code: ReconciliationIssueCode
    message: str
    position_id: str
    expected: float | str | None = None
    observed: float | str | None = None


@dataclass(frozen=True, slots=True)
class PositionLedger:
    """Immutable collection of managed positions and processed fill IDs."""

    positions: tuple[ManagedPosition, ...] = ()
    processed_fill_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        position_ids = [item.position_id for item in self.positions]
        if len(position_ids) != len(set(position_ids)):
            raise ValueError("Position IDs must be unique")
        fill_ids = tuple(_identifier(item, "processed_fill_id") for item in self.processed_fill_ids)
        if len(fill_ids) != len(set(fill_ids)):
            raise ValueError("Processed fill IDs must be unique")
        object.__setattr__(self, "processed_fill_ids", fill_ids)

    def get(self, position_id: str) -> ManagedPosition | None:
        normalized = position_id.strip()
        return next((item for item in self.positions if item.position_id == normalized), None)


@dataclass(frozen=True, slots=True)
class FillReconciliationResult:
    """Immutable result of applying one normalized broker fill."""

    status: ReconciliationStatus
    fill: BrokerFill
    ledger: PositionLedger
    position: ManagedPosition | None
    issues: tuple[ReconciliationIssue, ...] = ()

    @property
    def applied(self) -> bool:
        return self.status is ReconciliationStatus.APPLIED


class FillReconciler:
    """Apply broker fills exactly once and reject invalid state transitions."""

    def __init__(self, *, account_currency: str, converter: CurrencyConverter) -> None:
        currency = account_currency.strip().upper()
        if not currency:
            raise ValueError("account_currency cannot be empty")
        self.account_currency = currency
        self.converter = converter

    def apply(self, ledger: PositionLedger, fill: BrokerFill) -> FillReconciliationResult:
        """Apply one fill idempotently, returning a new ledger."""

        if fill.fill_id in ledger.processed_fill_ids:
            issue = ReconciliationIssue(
                code=ReconciliationIssueCode.DUPLICATE_FILL,
                message="Fill was already processed",
                position_id=fill.position_id,
            )
            return FillReconciliationResult(
                status=ReconciliationStatus.DUPLICATE,
                fill=fill,
                ledger=ledger,
                position=ledger.get(fill.position_id),
                issues=(issue,),
            )

        existing = ledger.get(fill.position_id)
        preflight_issue = self._preflight(existing, fill)
        if preflight_issue is not None:
            return FillReconciliationResult(
                status=ReconciliationStatus.REJECTED,
                fill=fill,
                ledger=ledger,
                position=existing,
                issues=(preflight_issue,),
            )

        try:
            if fill.action is FillAction.ENTRY:
                if existing is None:
                    updated = PositionLifecycle.open(
                        position_id=fill.position_id,
                        instrument=fill.instrument,
                        side=fill.position_side,
                        volume=fill.volume,
                        entry_price=fill.price,
                        opened_at=fill.filled_at,
                        stop_price=fill.stop_price,
                        take_profit_price=fill.take_profit_price,
                        commission_account=fill.commission_account,
                    )
                else:
                    updated = PositionLifecycle.add_entry(
                        existing,
                        volume=fill.volume,
                        price=fill.price,
                        filled_at=fill.filled_at,
                        commission_account=fill.commission_account,
                    )
            else:
                assert existing is not None
                updated = PositionLifecycle.reduce(
                    existing,
                    volume=fill.volume,
                    price=fill.price,
                    filled_at=fill.filled_at,
                    account_currency=self.account_currency,
                    converter=self.converter,
                    commission_account=fill.commission_account,
                    swap_account=fill.swap_account,
                )
        except (PositionLifecycleError, ValueError) as exc:
            issue = ReconciliationIssue(
                code=(
                    ReconciliationIssueCode.OVER_CLOSE
                    if fill.action is FillAction.EXIT
                    and existing is not None
                    and fill.volume > existing.open_volume
                    else ReconciliationIssueCode.INVALID_TRANSITION
                ),
                message=str(exc),
                position_id=fill.position_id,
            )
            return FillReconciliationResult(
                status=ReconciliationStatus.REJECTED,
                fill=fill,
                ledger=ledger,
                position=existing,
                issues=(issue,),
            )

        positions = tuple(
            updated if item.position_id == updated.position_id else item
            for item in ledger.positions
        )
        if existing is None:
            positions = (*positions, updated)
        new_ledger = PositionLedger(
            positions=positions,
            processed_fill_ids=(*ledger.processed_fill_ids, fill.fill_id),
        )
        return FillReconciliationResult(
            status=ReconciliationStatus.APPLIED,
            fill=fill,
            ledger=new_ledger,
            position=updated,
        )

    @staticmethod
    def _preflight(
        existing: ManagedPosition | None,
        fill: BrokerFill,
    ) -> ReconciliationIssue | None:
        if fill.action is FillAction.EXIT and existing is None:
            return ReconciliationIssue(
                code=ReconciliationIssueCode.UNKNOWN_POSITION,
                message="Exit fill references an unknown position",
                position_id=fill.position_id,
            )
        if existing is None:
            return None
        if existing.symbol != fill.instrument.symbol.symbol:
            return ReconciliationIssue(
                code=ReconciliationIssueCode.SYMBOL_MISMATCH,
                message="Fill symbol does not match the managed position",
                position_id=fill.position_id,
                expected=existing.symbol,
                observed=fill.instrument.symbol.symbol,
            )
        if existing.side is not fill.position_side:
            return ReconciliationIssue(
                code=ReconciliationIssueCode.SIDE_MISMATCH,
                message="Fill position side does not match the managed position",
                position_id=fill.position_id,
                expected=existing.side.value,
                observed=fill.position_side.value,
            )
        return None


class PositionReconciler:
    """Compare managed open positions with a broker position snapshot."""

    def audit(
        self,
        internal_positions: tuple[ManagedPosition, ...],
        broker_positions: tuple[BrokerPositionSnapshot, ...],
        *,
        volume_tolerance: float = 1e-9,
        price_tolerance: float = 1e-8,
    ) -> tuple[ReconciliationIssue, ...]:
        """Return every material mismatch without modifying either snapshot."""

        if volume_tolerance < 0.0 or price_tolerance < 0.0:
            raise ValueError("Reconciliation tolerances must be non-negative")
        internal = {item.position_id: item for item in internal_positions if item.open_volume > 0.0}
        broker = {item.position_id: item for item in broker_positions}
        issues: list[ReconciliationIssue] = []

        for position_id in sorted(internal.keys() - broker.keys()):
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.EXTRA_INTERNAL_POSITION,
                    message="Managed position is absent from broker snapshot",
                    position_id=position_id,
                )
            )
        for position_id in sorted(broker.keys() - internal.keys()):
            issues.append(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.MISSING_INTERNAL_POSITION,
                    message="Broker position is absent from internal state",
                    position_id=position_id,
                )
            )

        for position_id in sorted(internal.keys() & broker.keys()):
            expected = internal[position_id]
            observed = broker[position_id]
            if expected.symbol != observed.symbol:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.SYMBOL_MISMATCH,
                        message="Position symbol mismatch",
                        position_id=position_id,
                        expected=expected.symbol,
                        observed=observed.symbol,
                    )
                )
            if expected.side is not observed.side:
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.SIDE_MISMATCH,
                        message="Position side mismatch",
                        position_id=position_id,
                        expected=expected.side.value,
                        observed=observed.side.value,
                    )
                )
            if not isclose(
                expected.open_volume,
                observed.volume,
                rel_tol=0.0,
                abs_tol=volume_tolerance,
            ):
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.VOLUME_MISMATCH,
                        message="Position volume mismatch",
                        position_id=position_id,
                        expected=expected.open_volume,
                        observed=observed.volume,
                    )
                )
            if not isclose(
                expected.average_entry_price,
                observed.average_entry_price,
                rel_tol=0.0,
                abs_tol=price_tolerance,
            ):
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.ENTRY_PRICE_MISMATCH,
                        message="Average entry price mismatch",
                        position_id=position_id,
                        expected=expected.average_entry_price,
                        observed=observed.average_entry_price,
                    )
                )
            if not self._optional_price_equal(
                expected.stop_price,
                observed.stop_price,
                price_tolerance,
            ):
                issues.append(
                    ReconciliationIssue(
                        code=ReconciliationIssueCode.STOP_PRICE_MISMATCH,
                        message="Protective stop mismatch",
                        position_id=position_id,
                        expected=expected.stop_price,
                        observed=observed.stop_price,
                    )
                )
        return tuple(issues)

    @staticmethod
    def _optional_price_equal(
        expected: float | None,
        observed: float | None,
        tolerance: float,
    ) -> bool:
        if expected is None or observed is None:
            return expected is observed
        return isclose(expected, observed, rel_tol=0.0, abs_tol=tolerance)
