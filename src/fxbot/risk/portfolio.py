"""Immutable portfolio state and account-currency exposure aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from fxbot.risk.models import AccountSnapshot, InstrumentRiskSpec, TradeSide
from fxbot.risk.position_sizing import CurrencyConverter, convert_amount, risk_per_lot


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


def _validate_stop(side: TradeSide, entry_price: float, stop_price: float | None) -> None:
    if stop_price is None:
        return
    if side is TradeSide.LONG and stop_price >= entry_price:
        raise ValueError("A long exposure requires stop_price below entry_price")
    if side is TradeSide.SHORT and stop_price <= entry_price:
        raise ValueError("A short exposure requires stop_price above entry_price")


@dataclass(frozen=True, slots=True)
class PositionExposure:
    """One open position represented with all risk-relevant broker values."""

    position_id: str
    instrument: InstrumentRiskSpec
    side: TradeSide
    volume: float
    entry_price: float
    current_price: float
    stop_price: float | None
    margin_used: float
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_id", _identifier(self.position_id, "position_id"))
        object.__setattr__(self, "side", TradeSide(self.side))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(self, "entry_price", _positive(self.entry_price, "entry_price"))
        object.__setattr__(self, "current_price", _positive(self.current_price, "current_price"))
        object.__setattr__(self, "margin_used", _non_negative(self.margin_used, "margin_used"))
        object.__setattr__(self, "realized_pnl", _finite(self.realized_pnl, "realized_pnl"))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _positive(self.stop_price, "stop_price"))
        _validate_stop(self.side, self.entry_price, self.stop_price)

    @property
    def symbol(self) -> str:
        return self.instrument.symbol.symbol

    @property
    def signed_volume(self) -> float:
        return self.volume if self.side is TradeSide.LONG else -self.volume

    @property
    def unrealized_pnl_quote(self) -> float:
        direction = 1.0 if self.side is TradeSide.LONG else -1.0
        return (
            direction
            * (self.current_price - self.entry_price)
            * self.instrument.symbol.contract_size
            * self.volume
        )


@dataclass(frozen=True, slots=True)
class PendingOrderExposure:
    """One pending order whose potential fill consumes risk and margin capacity."""

    order_id: str
    instrument: InstrumentRiskSpec
    side: TradeSide
    volume: float
    entry_price: float
    stop_price: float | None
    margin_required: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _identifier(self.order_id, "order_id"))
        object.__setattr__(self, "side", TradeSide(self.side))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(self, "entry_price", _positive(self.entry_price, "entry_price"))
        object.__setattr__(
            self,
            "margin_required",
            _non_negative(self.margin_required, "margin_required"),
        )
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _positive(self.stop_price, "stop_price"))
        _validate_stop(self.side, self.entry_price, self.stop_price)

    @property
    def symbol(self) -> str:
        return self.instrument.symbol.symbol

    @property
    def signed_volume(self) -> float:
        return self.volume if self.side is TradeSide.LONG else -self.volume


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A market or pending order awaiting portfolio-level risk approval."""

    proposal_id: str
    instrument: InstrumentRiskSpec
    side: TradeSide
    volume: float
    entry_price: float
    stop_price: float | None
    margin_required: float
    pending: bool = False
    submitted_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _identifier(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "side", TradeSide(self.side))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(self, "entry_price", _positive(self.entry_price, "entry_price"))
        object.__setattr__(
            self,
            "margin_required",
            _non_negative(self.margin_required, "margin_required"),
        )
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _positive(self.stop_price, "stop_price"))
        if self.submitted_at is not None:
            object.__setattr__(
                self,
                "submitted_at",
                _utc(self.submitted_at, "submitted_at"),
            )
        _validate_stop(self.side, self.entry_price, self.stop_price)

    @property
    def symbol(self) -> str:
        return self.instrument.symbol.symbol

    def as_position(self) -> PositionExposure:
        """Represent an approved market proposal as a newly opened position."""

        return PositionExposure(
            position_id=f"proposal:{self.proposal_id}",
            instrument=self.instrument,
            side=self.side,
            volume=self.volume,
            entry_price=self.entry_price,
            current_price=self.entry_price,
            stop_price=self.stop_price,
            margin_used=self.margin_required,
        )

    def as_pending_order(self) -> PendingOrderExposure:
        """Represent a pending proposal as contingent portfolio exposure."""

        return PendingOrderExposure(
            order_id=f"proposal:{self.proposal_id}",
            instrument=self.instrument,
            side=self.side,
            volume=self.volume,
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            margin_required=self.margin_required,
        )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time account, position, and intraday risk state."""

    account: AccountSnapshot
    as_of: datetime
    day_start_equity: float
    intraday_peak_equity: float
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    last_loss_time: datetime | None = None
    manual_kill_switch: bool = False
    automatic_kill_switch: bool = False
    kill_switch_reason: str | None = None
    positions: tuple[PositionExposure, ...] = ()
    pending_orders: tuple[PendingOrderExposure, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        object.__setattr__(
            self,
            "day_start_equity",
            _positive(self.day_start_equity, "day_start_equity"),
        )
        object.__setattr__(
            self,
            "intraday_peak_equity",
            _positive(self.intraday_peak_equity, "intraday_peak_equity"),
        )
        object.__setattr__(
            self,
            "realized_pnl_today",
            _finite(self.realized_pnl_today, "realized_pnl_today"),
        )
        if self.consecutive_losses < 0:
            raise ValueError("consecutive_losses must be non-negative")
        if self.last_loss_time is not None:
            object.__setattr__(
                self,
                "last_loss_time",
                _utc(self.last_loss_time, "last_loss_time"),
            )
        reason = None if self.kill_switch_reason is None else self.kill_switch_reason.strip()
        object.__setattr__(self, "kill_switch_reason", reason or None)
        if (self.manual_kill_switch or self.automatic_kill_switch) and reason is None:
            object.__setattr__(self, "kill_switch_reason", "unspecified")
        _require_unique_identifiers(self.positions, self.pending_orders)


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Account-currency exposure and drawdown metrics for one portfolio state."""

    account_equity: float
    day_start_equity: float
    intraday_peak_equity: float
    open_position_count: int
    pending_order_count: int
    unprotected_exposure_count: int
    open_risk_amount: float
    pending_risk_amount: float
    total_risk_amount: float
    gross_notional_amount: float
    net_notional_amount: float
    margin_committed: float
    margin_utilization: float
    unrealized_pnl: float
    realized_pnl_today: float
    daily_realized_loss_amount: float
    daily_total_loss_amount: float
    daily_total_loss_fraction: float
    intraday_drawdown_amount: float
    intraday_drawdown_fraction: float
    equity_fraction_of_balance: float
    symbol_counts: tuple[tuple[str, int], ...]
    symbol_gross_notional: tuple[tuple[str, float], ...]
    currency_net_exposure: tuple[tuple[str, float], ...]

    def symbol_count(self, symbol: str) -> int:
        normalized = symbol.strip().upper()
        return dict(self.symbol_counts).get(normalized, 0)

    def symbol_notional(self, symbol: str) -> float:
        normalized = symbol.strip().upper()
        return dict(self.symbol_gross_notional).get(normalized, 0.0)

    def currency_exposure(self, currency: str) -> float:
        normalized = currency.strip().upper()
        return dict(self.currency_net_exposure).get(normalized, 0.0)

    @property
    def largest_currency_exposure(self) -> float:
        return max((abs(value) for _, value in self.currency_net_exposure), default=0.0)


class PortfolioAnalyzer:
    """Aggregate open and contingent exposure into account-currency metrics."""

    def __init__(self, converter: CurrencyConverter) -> None:
        self.converter = converter

    def analyze(
        self,
        snapshot: PortfolioSnapshot,
        *,
        additional_positions: Iterable[PositionExposure] = (),
        additional_pending_orders: Iterable[PendingOrderExposure] = (),
        include_pending_orders: bool = True,
    ) -> PortfolioMetrics:
        extra_positions = tuple(additional_positions)
        extra_pending_orders = tuple(additional_pending_orders)
        positions = (*snapshot.positions, *extra_positions)
        pending_orders = (*snapshot.pending_orders, *extra_pending_orders)

        symbol_counts: defaultdict[str, int] = defaultdict(int)
        symbol_notional: defaultdict[str, float] = defaultdict(float)
        native_currency_exposure: defaultdict[str, float] = defaultdict(float)
        open_risk = 0.0
        pending_risk = 0.0
        gross_notional = 0.0
        net_notional = 0.0
        existing_position_margin = sum(item.margin_used for item in snapshot.positions)
        margin_committed = max(snapshot.account.margin_used, existing_position_margin) + sum(
            item.margin_used for item in extra_positions
        )
        unrealized_pnl = 0.0
        unprotected = 0

        for position in positions:
            symbol_counts[position.symbol] += 1
            price = position.current_price
            notional = self._notional(
                position.instrument,
                position.volume,
                price,
                snapshot.account.currency,
            )
            signed_notional = notional if position.side is TradeSide.LONG else -notional
            gross_notional += abs(notional)
            net_notional += signed_notional
            symbol_notional[position.symbol] += abs(notional)
            self._add_currency_legs(
                native_currency_exposure,
                position.instrument,
                position.side,
                position.volume,
                price,
            )
            unrealized_pnl += convert_amount(
                abs(position.unrealized_pnl_quote),
                position.instrument.symbol.quote_currency,
                snapshot.account.currency,
                self.converter,
            ) * (1.0 if position.unrealized_pnl_quote >= 0.0 else -1.0)
            if position.stop_price is None:
                unprotected += 1
            else:
                open_risk += self._risk(
                    position.instrument,
                    position.volume,
                    position.entry_price,
                    position.stop_price,
                    snapshot.account.currency,
                )

        for order in pending_orders:
            symbol_counts[order.symbol] += 1
            if order.stop_price is None:
                unprotected += 1
            if not include_pending_orders:
                continue
            margin_committed += order.margin_required
            notional = self._notional(
                order.instrument,
                order.volume,
                order.entry_price,
                snapshot.account.currency,
            )
            signed_notional = notional if order.side is TradeSide.LONG else -notional
            gross_notional += abs(notional)
            net_notional += signed_notional
            symbol_notional[order.symbol] += abs(notional)
            self._add_currency_legs(
                native_currency_exposure,
                order.instrument,
                order.side,
                order.volume,
                order.entry_price,
            )
            if order.stop_price is not None:
                pending_risk += self._risk(
                    order.instrument,
                    order.volume,
                    order.entry_price,
                    order.stop_price,
                    snapshot.account.currency,
                )

        currency_exposure = tuple(
            sorted(
                (
                    currency,
                    self._signed_conversion(
                        amount,
                        currency,
                        snapshot.account.currency,
                    ),
                )
                for currency, amount in native_currency_exposure.items()
            )
        )
        daily_realized_loss = max(-snapshot.realized_pnl_today, 0.0)
        daily_total_loss = max(snapshot.day_start_equity - snapshot.account.equity, 0.0)
        drawdown = max(snapshot.intraday_peak_equity - snapshot.account.equity, 0.0)
        margin_utilization = margin_committed / snapshot.account.equity
        daily_total_loss_fraction = daily_total_loss / snapshot.day_start_equity
        drawdown_fraction = drawdown / snapshot.intraday_peak_equity
        equity_fraction = (
            snapshot.account.equity / snapshot.account.balance
            if snapshot.account.balance > 0.0
            else 1.0
        )

        return PortfolioMetrics(
            account_equity=snapshot.account.equity,
            day_start_equity=snapshot.day_start_equity,
            intraday_peak_equity=snapshot.intraday_peak_equity,
            open_position_count=len(positions),
            pending_order_count=len(pending_orders),
            unprotected_exposure_count=unprotected,
            open_risk_amount=open_risk,
            pending_risk_amount=pending_risk,
            total_risk_amount=open_risk + pending_risk,
            gross_notional_amount=gross_notional,
            net_notional_amount=net_notional,
            margin_committed=margin_committed,
            margin_utilization=margin_utilization,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_today=snapshot.realized_pnl_today,
            daily_realized_loss_amount=daily_realized_loss,
            daily_total_loss_amount=daily_total_loss,
            daily_total_loss_fraction=daily_total_loss_fraction,
            intraday_drawdown_amount=drawdown,
            intraday_drawdown_fraction=drawdown_fraction,
            equity_fraction_of_balance=equity_fraction,
            symbol_counts=tuple(sorted(symbol_counts.items())),
            symbol_gross_notional=tuple(sorted(symbol_notional.items())),
            currency_net_exposure=currency_exposure,
        )

    def _risk(
        self,
        instrument: InstrumentRiskSpec,
        volume: float,
        entry_price: float,
        stop_price: float,
        account_currency: str,
    ) -> float:
        per_lot = risk_per_lot(
            instrument,
            abs(entry_price - stop_price),
            account_currency,
            self.converter,
        )
        return per_lot * volume

    def _notional(
        self,
        instrument: InstrumentRiskSpec,
        volume: float,
        price: float,
        account_currency: str,
    ) -> float:
        quote_amount = instrument.symbol.contract_size * volume * price
        return convert_amount(
            quote_amount,
            instrument.symbol.quote_currency,
            account_currency,
            self.converter,
        )

    @staticmethod
    def _add_currency_legs(
        accumulator: defaultdict[str, float],
        instrument: InstrumentRiskSpec,
        side: TradeSide,
        volume: float,
        price: float,
    ) -> None:
        direction = 1.0 if side is TradeSide.LONG else -1.0
        base_units = instrument.symbol.contract_size * volume * direction
        quote_units = -base_units * price
        accumulator[instrument.symbol.base_currency] += base_units
        accumulator[instrument.symbol.quote_currency] += quote_units

    def _signed_conversion(
        self,
        amount: float,
        source_currency: str,
        account_currency: str,
    ) -> float:
        converted = convert_amount(
            abs(amount),
            source_currency,
            account_currency,
            self.converter,
        )
        return converted if amount >= 0.0 else -converted


def _require_unique_identifiers(
    positions: tuple[PositionExposure, ...],
    pending_orders: tuple[PendingOrderExposure, ...],
) -> None:
    identifiers = [item.position_id for item in positions]
    identifiers.extend(item.order_id for item in pending_orders)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Position and pending-order identifiers must be unique")
