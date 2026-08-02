"""Immutable position lifecycle, fills, and account-currency PnL valuation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import isclose, isfinite

from fxbot.risk.models import InstrumentRiskSpec, TradeSide
from fxbot.risk.portfolio import PositionExposure
from fxbot.risk.position_sizing import CurrencyConverter, convert_amount

_VOLUME_TOLERANCE = 1e-12


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


def _signed_conversion(
    amount: float,
    from_currency: str,
    to_currency: str,
    converter: CurrencyConverter,
) -> float:
    sign = 1.0 if amount >= 0.0 else -1.0
    return sign * convert_amount(abs(amount), from_currency, to_currency, converter)


def _validate_protective_prices(
    side: TradeSide,
    entry_price: float,
    stop_price: float | None,
    take_profit_price: float | None,
) -> None:
    if stop_price is not None:
        if side is TradeSide.LONG and stop_price >= entry_price:
            raise ValueError("A long position requires stop_price below entry_price")
        if side is TradeSide.SHORT and stop_price <= entry_price:
            raise ValueError("A short position requires stop_price above entry_price")
    if take_profit_price is not None:
        if side is TradeSide.LONG and take_profit_price <= entry_price:
            raise ValueError("A long position requires take_profit_price above entry_price")
        if side is TradeSide.SHORT and take_profit_price >= entry_price:
            raise ValueError("A short position requires take_profit_price below entry_price")


class PositionLifecycleError(RuntimeError):
    """Raised when a requested position transition is not valid."""


class PositionStatus(StrEnum):
    """Durable lifecycle state of a broker position."""

    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ManagedPosition:
    """Immutable, fully auditable state of one netted trading position."""

    position_id: str
    instrument: InstrumentRiskSpec
    side: TradeSide
    total_opened_volume: float
    open_volume: float
    average_entry_price: float
    initial_stop_price: float | None
    stop_price: float | None
    take_profit_price: float | None
    opened_at: datetime
    updated_at: datetime
    status: PositionStatus = PositionStatus.OPEN
    total_closed_volume: float = 0.0
    realized_pnl_account: float = 0.0
    commission_account: float = 0.0
    swap_account: float = 0.0
    closed_at: datetime | None = None
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_id", _identifier(self.position_id, "position_id"))
        object.__setattr__(self, "side", TradeSide(self.side))
        object.__setattr__(self, "status", PositionStatus(self.status))
        object.__setattr__(
            self,
            "total_opened_volume",
            _positive(self.total_opened_volume, "total_opened_volume"),
        )
        object.__setattr__(self, "open_volume", _non_negative(self.open_volume, "open_volume"))
        object.__setattr__(
            self,
            "total_closed_volume",
            _non_negative(self.total_closed_volume, "total_closed_volume"),
        )
        object.__setattr__(
            self,
            "average_entry_price",
            _positive(self.average_entry_price, "average_entry_price"),
        )
        for name in ("initial_stop_price", "stop_price", "take_profit_price"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))
        object.__setattr__(self, "opened_at", _utc(self.opened_at, "opened_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        if self.closed_at is not None:
            object.__setattr__(self, "closed_at", _utc(self.closed_at, "closed_at"))
        object.__setattr__(
            self,
            "realized_pnl_account",
            _finite(self.realized_pnl_account, "realized_pnl_account"),
        )
        object.__setattr__(
            self,
            "commission_account",
            _non_negative(self.commission_account, "commission_account"),
        )
        object.__setattr__(self, "swap_account", _finite(self.swap_account, "swap_account"))
        if self.updated_at < self.opened_at:
            raise ValueError("updated_at cannot be earlier than opened_at")
        if self.closed_at is not None and self.closed_at < self.opened_at:
            raise ValueError("closed_at cannot be earlier than opened_at")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if not isclose(
            self.open_volume + self.total_closed_volume,
            self.total_opened_volume,
            rel_tol=0.0,
            abs_tol=_VOLUME_TOLERANCE,
        ):
            raise ValueError(
                "open_volume plus total_closed_volume must equal total_opened_volume"
            )
        _validate_protective_prices(
            self.side,
            self.average_entry_price,
            self.initial_stop_price,
            self.take_profit_price,
        )
        # A current stop may legitimately be at or beyond break-even after the
        # position has moved into profit; only its positivity is enforced here.
        if self.status is PositionStatus.CLOSED:
            if self.open_volume > _VOLUME_TOLERANCE or self.closed_at is None:
                raise ValueError("Closed positions require zero open volume and closed_at")
        else:
            if self.open_volume <= _VOLUME_TOLERANCE:
                raise ValueError("Open positions require positive open_volume")
            if self.closed_at is not None:
                raise ValueError("Non-closed positions cannot define closed_at")
        if self.status is PositionStatus.OPEN and self.total_closed_volume > _VOLUME_TOLERANCE:
            raise ValueError("A position with closed volume must be partially closed")
        if (
            self.status is PositionStatus.PARTIALLY_CLOSED
            and self.total_closed_volume <= _VOLUME_TOLERANCE
        ):
            raise ValueError("Partially closed positions require closed volume")

    @property
    def symbol(self) -> str:
        return self.instrument.symbol.symbol

    @property
    def net_realized_pnl_account(self) -> float:
        """Realized trading PnL after commission and signed swap."""

        return self.realized_pnl_account - self.commission_account + self.swap_account

    @property
    def initial_risk_distance(self) -> float | None:
        if self.initial_stop_price is None:
            return None
        return abs(self.average_entry_price - self.initial_stop_price)

    def to_exposure(self, *, current_price: float, margin_used: float) -> PositionExposure:
        """Project lifecycle state into the Phase 2B portfolio-risk model."""

        if self.status is PositionStatus.CLOSED:
            raise PositionLifecycleError("A closed position has no open exposure")
        return PositionExposure(
            position_id=self.position_id,
            instrument=self.instrument,
            side=self.side,
            volume=self.open_volume,
            entry_price=self.average_entry_price,
            current_price=current_price,
            stop_price=self.stop_price,
            margin_used=margin_used,
            realized_pnl=self.net_realized_pnl_account,
        )


@dataclass(frozen=True, slots=True)
class PositionValuation:
    """Mark-to-market PnL for one managed position."""

    position_id: str
    market_price: float
    open_volume: float
    unrealized_pnl_quote: float
    unrealized_pnl_account: float
    realized_pnl_account: float
    net_realized_pnl_account: float
    total_pnl_account: float


class PositionLifecycle:
    """Pure immutable transitions for entries, reductions, and valuation."""

    @staticmethod
    def open(
        *,
        position_id: str,
        instrument: InstrumentRiskSpec,
        side: TradeSide,
        volume: float,
        entry_price: float,
        opened_at: datetime,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
        commission_account: float = 0.0,
    ) -> ManagedPosition:
        """Create a new position from its first accepted entry fill."""

        return ManagedPosition(
            position_id=position_id,
            instrument=instrument,
            side=side,
            total_opened_volume=volume,
            open_volume=volume,
            average_entry_price=entry_price,
            initial_stop_price=stop_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            opened_at=opened_at,
            updated_at=opened_at,
            commission_account=commission_account,
        )

    @staticmethod
    def add_entry(
        position: ManagedPosition,
        *,
        volume: float,
        price: float,
        filled_at: datetime,
        commission_account: float = 0.0,
    ) -> ManagedPosition:
        """Add same-direction volume and recalculate weighted average entry."""

        if position.status is PositionStatus.CLOSED:
            raise PositionLifecycleError("Cannot add volume to a closed position")
        fill_volume = _positive(volume, "volume")
        fill_price = _positive(price, "price")
        timestamp = _utc(filled_at, "filled_at")
        if timestamp < position.updated_at:
            raise PositionLifecycleError("Entry fill cannot predate the current position state")
        commission = _non_negative(commission_account, "commission_account")
        new_open = position.open_volume + fill_volume
        weighted_entry = (
            position.average_entry_price * position.open_volume + fill_price * fill_volume
        ) / new_open
        # Existing protective levels may become invalid after averaging. Reject
        # the fill rather than silently weakening or removing risk controls.
        _validate_protective_prices(
            position.side,
            weighted_entry,
            position.initial_stop_price,
            position.take_profit_price,
        )
        return replace(
            position,
            total_opened_volume=position.total_opened_volume + fill_volume,
            open_volume=new_open,
            average_entry_price=weighted_entry,
            updated_at=timestamp,
            commission_account=position.commission_account + commission,
            version=position.version + 1,
        )

    @staticmethod
    def reduce(
        position: ManagedPosition,
        *,
        volume: float,
        price: float,
        filled_at: datetime,
        account_currency: str,
        converter: CurrencyConverter,
        commission_account: float = 0.0,
        swap_account: float = 0.0,
    ) -> ManagedPosition:
        """Apply a partial or final exit and realize PnL in account currency."""

        if position.status is PositionStatus.CLOSED:
            raise PositionLifecycleError("Cannot reduce a closed position")
        exit_volume = _positive(volume, "volume")
        exit_price = _positive(price, "price")
        timestamp = _utc(filled_at, "filled_at")
        if timestamp < position.updated_at:
            raise PositionLifecycleError("Exit fill cannot predate the current position state")
        if exit_volume - position.open_volume > _VOLUME_TOLERANCE:
            raise PositionLifecycleError("Exit volume exceeds the open position volume")
        commission = _non_negative(commission_account, "commission_account")
        swap = _finite(swap_account, "swap_account")
        direction = 1.0 if position.side is TradeSide.LONG else -1.0
        pnl_quote = (
            direction
            * (exit_price - position.average_entry_price)
            * position.instrument.symbol.contract_size
            * exit_volume
        )
        pnl_account = _signed_conversion(
            pnl_quote,
            position.instrument.symbol.quote_currency,
            account_currency,
            converter,
        )
        remaining = max(position.open_volume - exit_volume, 0.0)
        closed_volume = position.total_closed_volume + exit_volume
        closed = remaining <= _VOLUME_TOLERANCE
        return replace(
            position,
            open_volume=0.0 if closed else remaining,
            total_closed_volume=closed_volume,
            realized_pnl_account=position.realized_pnl_account + pnl_account,
            commission_account=position.commission_account + commission,
            swap_account=position.swap_account + swap,
            updated_at=timestamp,
            closed_at=timestamp if closed else None,
            status=(
                PositionStatus.CLOSED if closed else PositionStatus.PARTIALLY_CLOSED
            ),
            version=position.version + 1,
        )

    @staticmethod
    def value(
        position: ManagedPosition,
        *,
        market_price: float,
        account_currency: str,
        converter: CurrencyConverter,
    ) -> PositionValuation:
        """Calculate current unrealized and cumulative account-currency PnL."""

        price = _positive(market_price, "market_price")
        direction = 1.0 if position.side is TradeSide.LONG else -1.0
        pnl_quote = (
            direction
            * (price - position.average_entry_price)
            * position.instrument.symbol.contract_size
            * position.open_volume
        )
        pnl_account = _signed_conversion(
            pnl_quote,
            position.instrument.symbol.quote_currency,
            account_currency,
            converter,
        )
        net_realized = position.net_realized_pnl_account
        return PositionValuation(
            position_id=position.position_id,
            market_price=price,
            open_volume=position.open_volume,
            unrealized_pnl_quote=pnl_quote,
            unrealized_pnl_account=pnl_account,
            realized_pnl_account=position.realized_pnl_account,
            net_realized_pnl_account=net_realized,
            total_pnl_account=net_realized + pnl_account,
        )
