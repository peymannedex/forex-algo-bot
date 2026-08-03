"""Immutable contracts for paper-runtime integration and acceptance testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from fxbot.execution.models import BrokerOrder, OrderIntent, Quote
from fxbot.execution.runtime import SyncResult
from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.models import StrategyDecision


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PaperFrame:
    """One point-in-time market context and executable quote."""

    context: MultiTimeframeContext
    quote: Quote

    def __post_init__(self) -> None:
        if self.context.symbol != self.quote.symbol:
            raise ValueError("context and quote symbols must match")
        if self.quote.timestamp != self.context.as_of:
            raise ValueError("quote timestamp must equal context as_of")


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """One netted paper position with account-currency realized PnL."""

    symbol: str
    signed_quantity: float
    average_price: float
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        quantity = float(self.signed_quantity)
        average_price = float(self.average_price)
        realized = float(self.realized_pnl)
        if not all(isfinite(value) for value in (quantity, average_price, realized)):
            raise ValueError("position values must be finite")
        if abs(quantity) <= 1e-12:
            raise ValueError("signed_quantity cannot be zero")
        if average_price <= 0.0:
            raise ValueError("average_price must be positive")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "signed_quantity", quantity)
        object.__setattr__(self, "average_price", average_price)
        object.__setattr__(self, "realized_pnl", realized)


@dataclass(frozen=True, slots=True)
class PaperAccountView:
    """Marked-to-market paper account summary."""

    currency: str
    balance: float
    equity: float
    day_start_equity: float
    peak_equity: float
    realized_pnl: float
    unrealized_pnl: float
    positions: tuple[PaperPosition, ...]
    updated_at: datetime

    def __post_init__(self) -> None:
        currency = self.currency.strip().upper()
        if not currency:
            raise ValueError("currency cannot be empty")
        for name in (
            "balance",
            "equity",
            "day_start_equity",
            "peak_equity",
            "realized_pnl",
            "unrealized_pnl",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.balance <= 0.0 or self.equity <= 0.0:
            raise ValueError("balance and equity must remain positive")
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))


@dataclass(frozen=True, slots=True)
class PaperOrderOutcome:
    """Order submission result that preserves non-fatal risk rejections."""

    intent: OrderIntent
    order: BrokerOrder | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.order is None) == (self.error is None):
            raise ValueError("exactly one of order or error is required")

    @property
    def accepted(self) -> bool:
        return self.order is not None


@dataclass(frozen=True, slots=True)
class PaperCycleResult:
    """Auditable result of one paper market frame."""

    cycle: int
    processed_at: datetime
    decision: StrategyDecision
    outcomes: tuple[PaperOrderOutcome, ...]
    sync: SyncResult
    account: PaperAccountView

    def __post_init__(self) -> None:
        if self.cycle < 1:
            raise ValueError("cycle must be positive")
        object.__setattr__(self, "processed_at", _utc(self.processed_at, "processed_at"))

    @property
    def accepted_orders(self) -> int:
        return sum(outcome.accepted for outcome in self.outcomes)

    @property
    def rejected_orders(self) -> int:
        return len(self.outcomes) - self.accepted_orders
