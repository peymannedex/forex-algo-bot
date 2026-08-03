from datetime import UTC, datetime, time, timedelta

from fxbot.execution.models import Quote
from fxbot.execution.safety import ExecutionControl
from fxbot.production.protections import (
    AccountRiskSnapshot,
    LossGuard,
    MarketHoursGuard,
    MarketWindow,
    QuoteGuard,
    QuoteProtectionConfig,
)


def quote(*, age: float = 0.0, bid: float = 1.1, ask: float = 1.1001) -> Quote:
    now = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    return Quote("EURUSD", bid, ask, now - timedelta(seconds=age))


def test_quote_guard_accepts_fresh_quote() -> None:
    decision = QuoteGuard().evaluate(
        quote(),
        now=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
    )

    assert decision.allowed


def test_quote_guard_rejects_stale_quote() -> None:
    guard = QuoteGuard(QuoteProtectionConfig(max_age_seconds=2, max_spread_bps=10))

    decision = guard.evaluate(
        quote(age=3),
        now=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
    )

    assert not decision.allowed
    assert decision.reason == "quote is stale"


def test_quote_guard_rejects_wide_spread() -> None:
    guard = QuoteGuard(QuoteProtectionConfig(max_age_seconds=2, max_spread_bps=5))

    decision = guard.evaluate(
        quote(bid=1.0, ask=1.01),
        now=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
    )

    assert not decision.allowed
    assert "spread" in decision.reason


def test_market_hours_normal_window() -> None:
    guard = MarketHoursGuard(
        (
            MarketWindow(
                frozenset({0, 1, 2, 3, 4}),
                time(7),
                time(20),
            ),
        )
    )

    assert guard.evaluate(datetime(2026, 1, 5, 12, tzinfo=UTC)).allowed
    assert not guard.evaluate(datetime(2026, 1, 5, 22, tzinfo=UTC)).allowed


def test_market_hours_overnight_window() -> None:
    guard = MarketHoursGuard(
        (MarketWindow(frozenset({0}), time(22), time(2)),)
    )

    assert guard.evaluate(datetime(2026, 1, 5, 23, tzinfo=UTC)).allowed
    assert guard.evaluate(datetime(2026, 1, 6, 1, tzinfo=UTC)).allowed


def test_loss_guard_trips_on_daily_loss() -> None:
    control = ExecutionControl.armed()
    guard = LossGuard(control, max_daily_loss=100, max_drawdown=200)
    snapshot = AccountRiskSnapshot(
        equity=900,
        daily_start_equity=1_000,
        peak_equity=1_000,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    decision = guard.evaluate(snapshot)

    assert not decision.allowed
    assert not control.state.enabled


def test_loss_guard_accepts_safe_account() -> None:
    control = ExecutionControl.armed()
    guard = LossGuard(control, max_daily_loss=100, max_drawdown=200)
    snapshot = AccountRiskSnapshot(
        equity=980,
        daily_start_equity=1_000,
        peak_equity=1_010,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert guard.evaluate(snapshot).allowed
    assert control.state.enabled
