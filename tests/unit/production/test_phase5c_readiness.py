from datetime import UTC, datetime

from fxbot.execution.connection import MT5ConnectionSnapshot
from fxbot.production.config import ProductionSettings
from fxbot.production.readiness import StartupReadinessGate


def snapshot(*, connected=True, trade_allowed=True):
    return MT5ConnectionSnapshot(
        connected=connected,
        account_login=123,
        server="Demo",
        trade_allowed=trade_allowed,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_paper_requires_dry_run() -> None:
    report = StartupReadinessGate(ProductionSettings()).evaluate(
        snapshot(),
        broker_dry_run=True,
    )

    assert report.ready


def test_paper_rejects_live_submission() -> None:
    report = StartupReadinessGate(ProductionSettings()).evaluate(
        snapshot(),
        broker_dry_run=False,
    )

    assert not report.ready


def test_demo_requires_trade_permission() -> None:
    settings = ProductionSettings(profile="demo")

    report = StartupReadinessGate(settings).evaluate(
        snapshot(trade_allowed=False),
        broker_dry_run=True,
    )

    assert not report.ready


def test_demo_can_submit_to_demo_account_when_enabled() -> None:
    settings = ProductionSettings(
        profile="demo",
        demo_order_submission_enabled=True,
    )

    report = StartupReadinessGate(settings).evaluate(
        snapshot(trade_allowed=True),
        broker_dry_run=False,
    )

    assert report.ready
