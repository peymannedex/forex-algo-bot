"""Backtest equity curve, audit ledger, and machine-readable result models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from fxbot.backtest.broker import BrokerSnapshot, ClosedTrade
from fxbot.backtest.config import BacktestConfig
from fxbot.backtest.events import AuditEvent, OrderState, OrderStatus, SimulatedFill


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Point-in-time account valuation used for return and drawdown analysis."""

    timestamp: datetime
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    unrealized_pnl: float
    drawdown_amount: float
    drawdown_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "timestamp"))
        for name in (
            "balance",
            "equity",
            "margin_used",
            "free_margin",
            "unrealized_pnl",
            "drawdown_amount",
            "drawdown_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.margin_used < 0.0 or self.drawdown_amount < 0.0:
            raise ValueError("margin_used and drawdown_amount cannot be negative")
        if not 0.0 <= self.drawdown_fraction <= 1.0:
            raise ValueError("drawdown_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete deterministic result of one event-driven simulation."""

    config: BacktestConfig
    started_at: datetime
    ended_at: datetime
    final_snapshot: BrokerSnapshot
    orders: tuple[OrderState, ...]
    fills: tuple[SimulatedFill, ...]
    trades: tuple[ClosedTrade, ...]
    equity_curve: tuple[EquityPoint, ...]
    audit_events: tuple[AuditEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "started_at", _utc(self.started_at, "started_at"))
        object.__setattr__(self, "ended_at", _utc(self.ended_at, "ended_at"))
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot predate started_at")
        previous: datetime | None = None
        for point in self.equity_curve:
            if previous is not None and point.timestamp < previous:
                raise ValueError("equity_curve must be chronologically ordered")
            previous = point.timestamp

    @property
    def initial_equity(self) -> float:
        return self.config.initial_cash

    @property
    def final_equity(self) -> float:
        return self.final_snapshot.equity

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_equity - 1.0

    @property
    def max_drawdown_fraction(self) -> float:
        return max((point.drawdown_fraction for point in self.equity_curve), default=0.0)

    @property
    def max_drawdown_amount(self) -> float:
        return max((point.drawdown_amount for point in self.equity_curve), default=0.0)

    @property
    def rejected_order_count(self) -> int:
        return sum(item.status is OrderStatus.REJECTED for item in self.orders)

    @property
    def cancelled_order_count(self) -> int:
        return sum(item.status is OrderStatus.CANCELLED for item in self.orders)

    @property
    def winning_trade_count(self) -> int:
        return sum(item.net_pnl > 0.0 for item in self.trades)

    @property
    def losing_trade_count(self) -> int:
        return sum(item.net_pnl < 0.0 for item in self.trades)

    @property
    def fingerprint(self) -> str:
        """Stable hash of financially material result fields."""

        payload = {
            "period": [self.started_at.isoformat(), self.ended_at.isoformat()],
            "seed": self.config.seed,
            "initial_cash": self.config.initial_cash,
            "final_equity": round(self.final_equity, 10),
            "fills": [
                [
                    item.fill_id,
                    item.order_id,
                    item.symbol,
                    item.side.value,
                    round(item.volume, 12),
                    round(item.price, 12),
                    item.timestamp.isoformat(),
                    round(item.commission, 12),
                ]
                for item in self.fills
            ],
            "trades": [
                [
                    item.trade_id,
                    item.symbol,
                    item.side.value,
                    round(item.volume, 12),
                    round(item.entry_price, 12),
                    round(item.exit_price, 12),
                    round(item.net_pnl, 10),
                ]
                for item in self.trades
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def equity_point(snapshot: BrokerSnapshot, running_peak: float) -> EquityPoint:
    """Build one equity point and update drawdown against the supplied peak."""

    peak = max(running_peak, snapshot.equity)
    drawdown = max(peak - snapshot.equity, 0.0)
    fraction = drawdown / peak if peak > 0.0 else 0.0
    return EquityPoint(
        timestamp=snapshot.timestamp,
        balance=snapshot.balance,
        equity=snapshot.equity,
        margin_used=snapshot.margin_used,
        free_margin=snapshot.free_margin,
        unrealized_pnl=snapshot.unrealized_pnl,
        drawdown_amount=drawdown,
        drawdown_fraction=fraction,
    )
