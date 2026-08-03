"""Pre-trade quote, market-hours, and account-loss protections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from math import isfinite

from fxbot.execution.models import Quote
from fxbot.execution.safety import ExecutionControl


@dataclass(frozen=True, slots=True)
class ProtectionDecision:
    allowed: bool
    reason: str
    metric: float | None = None
    limit: float | None = None


@dataclass(frozen=True, slots=True)
class QuoteProtectionConfig:
    max_age_seconds: float = 3.0
    max_spread_bps: float = 12.0

    def __post_init__(self) -> None:
        for name in ("max_age_seconds", "max_spread_bps"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


class QuoteGuard:
    """Reject stale or abnormally wide executable quotes."""

    def __init__(self, config: QuoteProtectionConfig | None = None) -> None:
        self.config = config or QuoteProtectionConfig()

    def evaluate(self, quote: Quote, *, now: datetime | None = None) -> ProtectionDecision:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        age = (current - quote.timestamp).total_seconds()
        if age < -1.0:
            return ProtectionDecision(False, "quote timestamp is in the future", age, 0.0)
        if age > self.config.max_age_seconds:
            return ProtectionDecision(
                False,
                "quote is stale",
                age,
                self.config.max_age_seconds,
            )
        spread_bps = 0.0 if quote.mid == 0.0 else quote.spread / quote.mid * 10_000.0
        if spread_bps > self.config.max_spread_bps:
            return ProtectionDecision(
                False,
                "spread exceeds configured limit",
                spread_bps,
                self.config.max_spread_bps,
            )
        return ProtectionDecision(True, "quote accepted", spread_bps, self.config.max_spread_bps)


@dataclass(frozen=True, slots=True)
class MarketWindow:
    """UTC trading window; an end before start represents an overnight window."""

    weekdays: frozenset[int]
    start: time
    end: time

    def __post_init__(self) -> None:
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekdays must contain integers from 0 through 6")

    def contains(self, timestamp: datetime) -> bool:
        current = timestamp.astimezone(UTC)
        current_time = current.time().replace(tzinfo=None)
        if self.start <= self.end:
            return current.weekday() in self.weekdays and self.start <= current_time <= self.end
        previous_day = (current.weekday() - 1) % 7
        return (
            current.weekday() in self.weekdays and current_time >= self.start
        ) or (
            previous_day in self.weekdays and current_time <= self.end
        )


class MarketHoursGuard:
    def __init__(self, windows: tuple[MarketWindow, ...]) -> None:
        if not windows:
            raise ValueError("at least one market window is required")
        self.windows = windows

    def evaluate(self, timestamp: datetime) -> ProtectionDecision:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        allowed = any(window.contains(timestamp) for window in self.windows)
        return ProtectionDecision(allowed, "market open" if allowed else "market closed")


@dataclass(frozen=True, slots=True)
class AccountRiskSnapshot:
    equity: float
    daily_start_equity: float
    peak_equity: float
    checked_at: datetime

    def __post_init__(self) -> None:
        for name in ("equity", "daily_start_equity", "peak_equity"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.checked_at.tzinfo is None or self.checked_at.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        object.__setattr__(self, "checked_at", self.checked_at.astimezone(UTC))

    @property
    def daily_loss(self) -> float:
        return max(self.daily_start_equity - self.equity, 0.0)

    @property
    def drawdown(self) -> float:
        return max(self.peak_equity - self.equity, 0.0)


class LossGuard:
    """Trip the shared execution control on daily-loss or drawdown breaches."""

    def __init__(
        self,
        control: ExecutionControl,
        *,
        max_daily_loss: float,
        max_drawdown: float,
    ) -> None:
        daily_limit = float(max_daily_loss)
        drawdown_limit = float(max_drawdown)
        if not isfinite(daily_limit) or daily_limit <= 0.0:
            raise ValueError("max_daily_loss must be positive and finite")
        if not isfinite(drawdown_limit) or drawdown_limit <= 0.0:
            raise ValueError("max_drawdown must be positive and finite")
        self.max_daily_loss = daily_limit
        self.max_drawdown = drawdown_limit
        self.control = control

    def evaluate(self, snapshot: AccountRiskSnapshot) -> ProtectionDecision:
        if snapshot.daily_loss >= self.max_daily_loss:
            reason = (
                f"daily loss limit breached: {snapshot.daily_loss:.2f} "
                f">= {self.max_daily_loss:.2f}"
            )
            self.control.trip(reason)
            return ProtectionDecision(
                False,
                reason,
                snapshot.daily_loss,
                self.max_daily_loss,
            )
        if snapshot.drawdown >= self.max_drawdown:
            reason = (
                f"drawdown limit breached: {snapshot.drawdown:.2f} "
                f">= {self.max_drawdown:.2f}"
            )
            self.control.trip(reason)
            return ProtectionDecision(
                False,
                reason,
                snapshot.drawdown,
                self.max_drawdown,
            )
        return ProtectionDecision(True, "account risk accepted")
