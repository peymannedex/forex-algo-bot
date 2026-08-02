"""Drawdown episode detection, duration, depth, and recovery analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fxbot.backtest.results import EquityPoint


@dataclass(frozen=True, slots=True)
class DrawdownPeriod:
    """One peak-to-trough-to-recovery episode in the equity curve."""

    started_at: datetime
    trough_at: datetime
    recovered_at: datetime | None
    peak_equity: float
    trough_equity: float
    amount: float
    fraction: float
    duration_seconds: float
    recovery_seconds: float | None


@dataclass(frozen=True, slots=True)
class DrawdownAnalysis:
    """Complete set of drawdown episodes and principal risk statistics."""

    periods: tuple[DrawdownPeriod, ...]
    maximum_amount: float
    maximum_fraction: float
    maximum_duration_seconds: float
    maximum_recovery_seconds: float | None
    unrecovered: bool

    @property
    def maximum_period(self) -> DrawdownPeriod | None:
        return max(self.periods, key=lambda item: (item.fraction, item.amount), default=None)


def analyze_drawdowns(equity_curve: tuple[EquityPoint, ...]) -> DrawdownAnalysis:
    """Detect every drawdown episode from a chronological equity curve."""

    if not equity_curve:
        return DrawdownAnalysis((), 0.0, 0.0, 0.0, None, False)
    _validate_curve(equity_curve)

    peak_equity = equity_curve[0].equity
    peak_time = equity_curve[0].timestamp
    active_start: datetime | None = None
    active_peak = peak_equity
    trough_equity = peak_equity
    trough_time = peak_time
    periods: list[DrawdownPeriod] = []

    for point in equity_curve[1:]:
        if active_start is None:
            if point.equity >= peak_equity:
                peak_equity = point.equity
                peak_time = point.timestamp
                continue
            active_start = peak_time
            active_peak = peak_equity
            trough_equity = point.equity
            trough_time = point.timestamp
            continue

        if point.equity < trough_equity:
            trough_equity = point.equity
            trough_time = point.timestamp
        if point.equity >= active_peak:
            periods.append(
                _period(
                    active_start,
                    trough_time,
                    point.timestamp,
                    active_peak,
                    trough_equity,
                    point.timestamp,
                )
            )
            peak_equity = point.equity
            peak_time = point.timestamp
            active_start = None

    if active_start is not None:
        periods.append(
            _period(
                active_start,
                trough_time,
                None,
                active_peak,
                trough_equity,
                equity_curve[-1].timestamp,
            )
        )

    maximum_amount = max((item.amount for item in periods), default=0.0)
    maximum_fraction = max((item.fraction for item in periods), default=0.0)
    maximum_duration = max((item.duration_seconds for item in periods), default=0.0)
    recovered_durations = tuple(
        item.recovery_seconds for item in periods if item.recovery_seconds is not None
    )
    return DrawdownAnalysis(
        periods=tuple(periods),
        maximum_amount=maximum_amount,
        maximum_fraction=maximum_fraction,
        maximum_duration_seconds=maximum_duration,
        maximum_recovery_seconds=(
            max(recovered_durations) if recovered_durations else None
        ),
        unrecovered=bool(periods and periods[-1].recovered_at is None),
    )


def _period(
    started_at: datetime,
    trough_at: datetime,
    recovered_at: datetime | None,
    peak_equity: float,
    trough_equity: float,
    terminal_time: datetime,
) -> DrawdownPeriod:
    amount = max(peak_equity - trough_equity, 0.0)
    fraction = amount / peak_equity if peak_equity > 0.0 else 0.0
    duration = (terminal_time - started_at).total_seconds()
    recovery = (
        (recovered_at - trough_at).total_seconds() if recovered_at is not None else None
    )
    return DrawdownPeriod(
        started_at=started_at,
        trough_at=trough_at,
        recovered_at=recovered_at,
        peak_equity=peak_equity,
        trough_equity=trough_equity,
        amount=amount,
        fraction=fraction,
        duration_seconds=duration,
        recovery_seconds=recovery,
    )


def _validate_curve(equity_curve: tuple[EquityPoint, ...]) -> None:
    previous = equity_curve[0].timestamp
    for point in equity_curve[1:]:
        if point.timestamp < previous:
            raise ValueError("equity_curve must be chronologically ordered")
        previous = point.timestamp
