"""Bid/ask-aware deterministic simulated broker and net position book."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from math import isfinite
from random import Random

from fxbot.backtest.config import BacktestConfig
from fxbot.backtest.events import (
    AuditEvent,
    EventKind,
    MarketDataRecord,
    MarketEvent,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    SimulatedFill,
    TimeInForce,
)
from fxbot.domain.models import Bar, Tick

_EPSILON = 1e-12


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class NetPosition:
    """Netted open exposure for one symbol."""

    symbol: str
    signed_volume: float
    average_entry_price: float
    opened_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        object.__setattr__(self, "symbol", symbol)
        if not isfinite(self.signed_volume) or abs(self.signed_volume) <= _EPSILON:
            raise ValueError("signed_volume must be finite and non-zero")
        object.__setattr__(
            self,
            "average_entry_price",
            _positive(self.average_entry_price, "average_entry_price"),
        )
        object.__setattr__(self, "opened_at", _utc(self.opened_at, "opened_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        if self.updated_at < self.opened_at:
            raise ValueError("updated_at cannot predate opened_at")

    @property
    def side(self) -> OrderSide:
        return OrderSide.BUY if self.signed_volume > 0.0 else OrderSide.SELL

    @property
    def volume(self) -> float:
        return abs(self.signed_volume)


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """One realized reduction of a net position."""

    trade_id: str
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    gross_pnl: float
    commission: float
    net_pnl: float


@dataclass(frozen=True, slots=True)
class BrokerSnapshot:
    """Point-in-time simulated account and position state."""

    timestamp: datetime
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    unrealized_pnl: float
    realized_pnl: float
    commissions: float
    swap: float
    positions: tuple[NetPosition, ...]
    pending_orders: tuple[OrderState, ...]

    def position(self, symbol: str) -> NetPosition | None:
        normalized = symbol.strip().upper()
        return next((item for item in self.positions if item.symbol == normalized), None)


@dataclass(frozen=True, slots=True)
class _QuoteView:
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float

    @property
    def mid_open(self) -> float:
        return (self.bid_open + self.ask_open) / 2.0

    @property
    def spread_bps(self) -> float:
        if self.mid_open <= 0.0:
            return 0.0
        return (self.ask_open - self.bid_open) / self.mid_open * 10_000.0


class SimulatedBroker:
    """Execute orders against historical bid/ask ticks and bars.

    Orders submitted after processing market event ``N`` become eligible only
    on event ``N+1`` or later. This prevents same-bar look-ahead fills.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._rng = Random(config.seed)
        self._cash = config.initial_cash
        self._realized_pnl = 0.0
        self._commissions = 0.0
        self._swap = 0.0
        self._orders: dict[str, OrderState] = {}
        self._positions: dict[str, NetPosition] = {}
        self._fills: list[SimulatedFill] = []
        self._trades: list[ClosedTrade] = []
        self._audit: list[AuditEvent] = []
        self._last_records: dict[str, MarketDataRecord] = {}
        self._last_time: datetime | None = None
        self._fill_counter = 0
        self._trade_counter = 0

    @property
    def orders(self) -> tuple[OrderState, ...]:
        return tuple(self._orders[key] for key in sorted(self._orders))

    @property
    def fills(self) -> tuple[SimulatedFill, ...]:
        return tuple(self._fills)

    @property
    def trades(self) -> tuple[ClosedTrade, ...]:
        return tuple(self._trades)

    @property
    def audit_events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit)

    def submit(self, request: OrderRequest, *, current_sequence: int) -> OrderState:
        """Accept a validated request into the pending order book."""

        if request.order_id in self._orders:
            raise ValueError(f"Duplicate order_id: {request.order_id}")
        self.config.instrument(request.symbol)
        if current_sequence < 0:
            raise ValueError("current_sequence must be non-negative")
        state = OrderState(
            request=request,
            status=OrderStatus.PENDING,
            remaining_volume=request.volume,
            filled_volume=0.0,
            average_fill_price=None,
            activated_after_sequence=current_sequence,
            updated_at=request.submitted_at,
        )
        self._orders[request.order_id] = state
        self._audit.append(
            AuditEvent(
                sequence=current_sequence,
                timestamp=request.submitted_at,
                kind=EventKind.ORDER_ACCEPTED,
                message="Order accepted by simulated broker",
                order_id=request.order_id,
            )
        )
        return state

    def reject(
        self,
        request: OrderRequest,
        *,
        current_sequence: int,
        reason: str,
    ) -> OrderState:
        """Record an order rejected before entering the execution book."""

        if request.order_id in self._orders:
            raise ValueError(f"Duplicate order_id: {request.order_id}")
        state = OrderState(
            request=request,
            status=OrderStatus.REJECTED,
            remaining_volume=request.volume,
            filled_volume=0.0,
            average_fill_price=None,
            activated_after_sequence=current_sequence,
            updated_at=request.submitted_at,
            rejection_reason=reason.strip() or "rejected",
        )
        self._orders[request.order_id] = state
        self._audit.append(
            AuditEvent(
                sequence=current_sequence,
                timestamp=request.submitted_at,
                kind=EventKind.ORDER_REJECTED,
                message=state.rejection_reason or "Order rejected",
                order_id=request.order_id,
            )
        )
        return state

    def cancel(
        self,
        order_id: str,
        *,
        timestamp: datetime,
        sequence: int,
        reason: str = "cancelled",
    ) -> OrderState:
        """Cancel an active pending or partially filled order."""

        try:
            current = self._orders[order_id]
        except KeyError as exc:
            raise KeyError(f"Unknown order_id: {order_id}") from exc
        if current.status.terminal:
            return current
        updated = replace(
            current,
            status=OrderStatus.CANCELLED,
            updated_at=_utc(timestamp, "timestamp"),
            rejection_reason=reason.strip() or None,
        )
        self._orders[order_id] = updated
        self._audit.append(
            AuditEvent(
                sequence=sequence,
                timestamp=updated.updated_at,
                kind=EventKind.ORDER_CANCELLED,
                message=reason.strip() or "Order cancelled",
                order_id=order_id,
            )
        )
        return updated

    def on_market(self, event: MarketEvent) -> tuple[SimulatedFill, ...]:
        """Advance broker time, accrue swap, and process eligible orders."""

        self._accrue_swap(event.timestamp, event.sequence)
        self._last_records[event.symbol] = event.record
        self._last_time = event.timestamp
        quote = self._quote(event.record)
        fills: list[SimulatedFill] = []

        active_ids = sorted(
            order_id
            for order_id, state in self._orders.items()
            if not state.status.terminal
            and state.request.symbol == event.symbol
            and event.sequence > state.activated_after_sequence
        )
        for order_id in active_ids:
            state = self._orders[order_id]
            if self._reject_for_spread(state, quote, event):
                continue
            if self.config.execution.rejection_probability > 0.0 and (
                self._rng.random() < self.config.execution.rejection_probability
            ):
                self._set_rejected(state, event, "stochastic_execution_rejection")
                continue

            base_price = self._trigger_price(state.request, quote)
            if base_price is None:
                if state.request.time_in_force is TimeInForce.IOC:
                    self.cancel(
                        order_id,
                        timestamp=event.timestamp,
                        sequence=event.sequence,
                        reason="ioc_not_filled",
                    )
                continue

            executable_volume = self._executable_volume(state)
            if executable_volume <= _EPSILON:
                self._set_rejected(state, event, "reduce_only_has_no_exposure")
                continue
            limit = self.config.execution.max_fill_volume_per_event
            fill_volume = executable_volume if limit is None else min(executable_volume, limit)
            fill = self._create_fill(state, event, base_price, fill_volume)
            self._apply_fill(fill, state.request.reduce_only)
            self._advance_order(state, fill)
            fills.append(fill)

        return tuple(fills)

    def liquidate(self, *, timestamp: datetime, sequence: int) -> tuple[SimulatedFill, ...]:
        """Close every open position at the last executable bid or ask."""

        timestamp = _utc(timestamp, "timestamp")
        fills: list[SimulatedFill] = []
        for symbol in sorted(tuple(self._positions)):
            position = self._positions.get(symbol)
            record = self._last_records.get(symbol)
            if position is None or record is None:
                continue
            side = OrderSide.SELL if position.signed_volume > 0.0 else OrderSide.BUY
            request = OrderRequest(
                order_id=f"liquidate:{symbol}:{sequence}",
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                volume=position.volume,
                submitted_at=timestamp,
                reduce_only=True,
                client_tag="liquidation",
            )
            state = OrderState(
                request=request,
                status=OrderStatus.PENDING,
                remaining_volume=request.volume,
                filled_volume=0.0,
                average_fill_price=None,
                activated_after_sequence=max(sequence - 1, 0),
                updated_at=timestamp,
            )
            self._orders[request.order_id] = state
            quote = self._quote(record)
            base_price = quote.bid_close if side is OrderSide.SELL else quote.ask_close
            fill = self._create_fill_at(
                state,
                sequence=sequence,
                timestamp=timestamp,
                base_price=base_price,
                volume=position.volume,
            )
            self._apply_fill(fill, True)
            self._advance_order(state, fill)
            fills.append(fill)
        self._last_time = timestamp
        return tuple(fills)

    def snapshot(self, timestamp: datetime | None = None) -> BrokerSnapshot:
        """Mark every open position to the executable close side."""

        effective_time = timestamp or self._last_time
        if effective_time is None:
            effective_time = datetime(1970, 1, 1, tzinfo=UTC)
        effective_time = _utc(effective_time, "timestamp")
        unrealized = 0.0
        margin = 0.0
        for position in self._positions.values():
            record = self._last_records.get(position.symbol)
            if record is None:
                continue
            quote = self._quote(record)
            mark = quote.bid_close if position.signed_volume > 0.0 else quote.ask_close
            instrument = self.config.instrument(position.symbol)
            unrealized += (
                (mark - position.average_entry_price)
                * instrument.contract_size
                * position.signed_volume
            )
            margin += (
                position.volume
                * instrument.contract_size
                * quote.mid_open
                / instrument.leverage
            )
        equity = self._cash + unrealized
        pending = tuple(item for item in self.orders if not item.status.terminal)
        return BrokerSnapshot(
            timestamp=effective_time,
            balance=self._cash,
            equity=equity,
            margin_used=margin,
            free_margin=equity - margin,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            commissions=self._commissions,
            swap=self._swap,
            positions=tuple(self._positions[key] for key in sorted(self._positions)),
            pending_orders=pending,
        )

    def _reject_for_spread(
        self,
        state: OrderState,
        quote: _QuoteView,
        event: MarketEvent,
    ) -> bool:
        maximum = self.config.execution.max_spread_bps
        if maximum is None or quote.spread_bps <= maximum:
            return False
        self._set_rejected(state, event, "spread_limit_exceeded")
        return True

    def _set_rejected(self, state: OrderState, event: MarketEvent, reason: str) -> None:
        updated = replace(
            state,
            status=OrderStatus.REJECTED,
            updated_at=event.timestamp,
            rejection_reason=reason,
        )
        self._orders[state.request.order_id] = updated
        self._audit.append(
            AuditEvent(
                sequence=event.sequence,
                timestamp=event.timestamp,
                kind=EventKind.ORDER_REJECTED,
                message=reason,
                order_id=state.request.order_id,
            )
        )

    def _executable_volume(self, state: OrderState) -> float:
        requested = state.remaining_volume
        if not state.request.reduce_only:
            if not self.config.allow_short and state.request.side is OrderSide.SELL:
                position = self._positions.get(state.request.symbol)
                if position is None or position.signed_volume <= 0.0:
                    return 0.0
                return min(requested, position.volume)
            return requested

        position = self._positions.get(state.request.symbol)
        if position is None:
            return 0.0
        if position.signed_volume > 0.0 and state.request.side is not OrderSide.SELL:
            return 0.0
        if position.signed_volume < 0.0 and state.request.side is not OrderSide.BUY:
            return 0.0
        return min(requested, position.volume)

    def _create_fill(
        self,
        state: OrderState,
        event: MarketEvent,
        base_price: float,
        volume: float,
    ) -> SimulatedFill:
        return self._create_fill_at(
            state,
            sequence=event.sequence,
            timestamp=event.timestamp,
            base_price=base_price,
            volume=volume,
        )

    def _create_fill_at(
        self,
        state: OrderState,
        *,
        sequence: int,
        timestamp: datetime,
        base_price: float,
        volume: float,
    ) -> SimulatedFill:
        self._fill_counter += 1
        jitter = self.config.execution.slippage.jitter_bps
        random_component = self._rng.uniform(-jitter, jitter) if jitter > 0.0 else 0.0
        total_bps = max(self.config.execution.slippage.base_bps + random_component, 0.0)
        slippage = base_price * total_bps / 10_000.0
        price = base_price + slippage if state.request.side is OrderSide.BUY else base_price - slippage
        commission = self.config.commission.calculate(volume)
        fill = SimulatedFill(
            fill_id=f"fill-{self._fill_counter:08d}",
            order_id=state.request.order_id,
            symbol=state.request.symbol,
            side=state.request.side,
            volume=volume,
            price=price,
            timestamp=timestamp,
            sequence=sequence,
            commission=commission,
            slippage_amount=slippage,
        )
        self._fills.append(fill)
        self._audit.append(
            AuditEvent(
                sequence=sequence,
                timestamp=fill.timestamp,
                kind=EventKind.FILL,
                message="Simulated order fill",
                order_id=fill.order_id,
                fill_id=fill.fill_id,
                metadata=(
                    ("price", f"{fill.price:.12g}"),
                    ("volume", f"{fill.volume:.12g}"),
                ),
            )
        )
        return fill

    def _advance_order(self, state: OrderState, fill: SimulatedFill) -> None:
        new_filled = state.filled_volume + fill.volume
        remaining = max(state.request.volume - new_filled, 0.0)
        if state.average_fill_price is None:
            average = fill.price
        else:
            average = (
                state.average_fill_price * state.filled_volume + fill.price * fill.volume
            ) / new_filled
        status = OrderStatus.FILLED if remaining <= _EPSILON else OrderStatus.PARTIALLY_FILLED
        self._orders[state.request.order_id] = replace(
            state,
            status=status,
            remaining_volume=0.0 if status is OrderStatus.FILLED else remaining,
            filled_volume=new_filled,
            average_fill_price=average,
            updated_at=fill.timestamp,
        )

    def _apply_fill(self, fill: SimulatedFill, reduce_only: bool) -> None:
        instrument = self.config.instrument(fill.symbol)
        signed_fill = fill.side.sign * fill.volume
        existing = self._positions.get(fill.symbol)
        gross_realized = 0.0

        if existing is None:
            if reduce_only:
                raise RuntimeError("reduce-only fill cannot open a position")
            self._positions[fill.symbol] = NetPosition(
                symbol=fill.symbol,
                signed_volume=signed_fill,
                average_entry_price=fill.price,
                opened_at=fill.timestamp,
                updated_at=fill.timestamp,
            )
        elif existing.signed_volume * signed_fill > 0.0:
            new_signed = existing.signed_volume + signed_fill
            weighted = (
                existing.average_entry_price * existing.volume + fill.price * fill.volume
            ) / abs(new_signed)
            self._positions[fill.symbol] = replace(
                existing,
                signed_volume=new_signed,
                average_entry_price=weighted,
                updated_at=fill.timestamp,
            )
        else:
            closed_volume = min(existing.volume, fill.volume)
            gross_realized = (
                (fill.price - existing.average_entry_price)
                * instrument.contract_size
                * closed_volume
                * (1.0 if existing.signed_volume > 0.0 else -1.0)
            )
            self._trade_counter += 1
            self._trades.append(
                ClosedTrade(
                    trade_id=f"trade-{self._trade_counter:08d}",
                    symbol=fill.symbol,
                    side=existing.side,
                    volume=closed_volume,
                    entry_price=existing.average_entry_price,
                    exit_price=fill.price,
                    opened_at=existing.opened_at,
                    closed_at=fill.timestamp,
                    gross_pnl=gross_realized,
                    commission=fill.commission,
                    net_pnl=gross_realized - fill.commission,
                )
            )
            remaining_signed = existing.signed_volume + signed_fill
            if abs(remaining_signed) <= _EPSILON:
                del self._positions[fill.symbol]
            elif existing.signed_volume * remaining_signed > 0.0:
                self._positions[fill.symbol] = replace(
                    existing,
                    signed_volume=remaining_signed,
                    updated_at=fill.timestamp,
                )
            else:
                if reduce_only:
                    raise RuntimeError("reduce-only fill cannot reverse a position")
                self._positions[fill.symbol] = NetPosition(
                    symbol=fill.symbol,
                    signed_volume=remaining_signed,
                    average_entry_price=fill.price,
                    opened_at=fill.timestamp,
                    updated_at=fill.timestamp,
                )

        self._cash += gross_realized - fill.commission
        self._realized_pnl += gross_realized
        self._commissions += fill.commission

    def _accrue_swap(self, current: datetime, sequence: int) -> None:
        if self._last_time is None or not self._positions:
            return
        previous = self._last_time
        for rollover in self._rollovers_between(previous, current):
            multiplier = 3.0 if rollover.weekday() == self.config.swap.triple_swap_weekday else 1.0
            amount = 0.0
            for position in self._positions.values():
                rate = (
                    self.config.swap.long_per_lot
                    if position.signed_volume > 0.0
                    else self.config.swap.short_per_lot
                )
                amount += rate * position.volume * multiplier
            if amount == 0.0:
                continue
            self._cash += amount
            self._swap += amount
            self._audit.append(
                AuditEvent(
                    sequence=sequence,
                    timestamp=rollover,
                    kind=EventKind.SWAP,
                    message="Daily swap applied",
                    metadata=(("amount", f"{amount:.12g}"),),
                )
            )

    def _rollovers_between(self, start: datetime, end: datetime) -> tuple[datetime, ...]:
        start = _utc(start, "start")
        end = _utc(end, "end")
        if end <= start:
            return ()
        current_date: date = start.date() - timedelta(days=1)
        last_date = end.date()
        result: list[datetime] = []
        while current_date <= last_date:
            rollover = datetime.combine(
                current_date,
                time(self.config.swap.rollover_hour_utc, tzinfo=UTC),
            )
            if start < rollover <= end:
                result.append(rollover)
            current_date += timedelta(days=1)
        return tuple(result)

    @staticmethod
    def _quote(record: MarketDataRecord) -> _QuoteView:
        if isinstance(record, Tick):
            return _QuoteView(
                bid_open=record.bid,
                bid_high=record.bid,
                bid_low=record.bid,
                bid_close=record.bid,
                ask_open=record.ask,
                ask_high=record.ask,
                ask_low=record.ask,
                ask_close=record.ask,
            )
        assert isinstance(record, Bar)
        return _QuoteView(
            bid_open=record.bid.open,
            bid_high=record.bid.high,
            bid_low=record.bid.low,
            bid_close=record.bid.close,
            ask_open=record.ask.open,
            ask_high=record.ask.high,
            ask_low=record.ask.low,
            ask_close=record.ask.close,
        )

    @staticmethod
    def _trigger_price(request: OrderRequest, quote: _QuoteView) -> float | None:
        if request.order_type is OrderType.MARKET:
            return quote.ask_open if request.side is OrderSide.BUY else quote.bid_open
        if request.order_type is OrderType.LIMIT:
            assert request.limit_price is not None
            if request.side is OrderSide.BUY:
                if quote.ask_low > request.limit_price:
                    return None
                return min(request.limit_price, quote.ask_open)
            if quote.bid_high < request.limit_price:
                return None
            return max(request.limit_price, quote.bid_open)
        assert request.stop_price is not None
        if request.side is OrderSide.BUY:
            if quote.ask_high < request.stop_price:
                return None
            return max(request.stop_price, quote.ask_open)
        if quote.bid_low > request.stop_price:
            return None
        return min(request.stop_price, quote.bid_open)
