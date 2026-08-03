from datetime import UTC, datetime

from fxbot.production.alerts import (
    AlertSeverity,
    FanoutAlertSink,
    InMemoryAlertSink,
    alert,
)


def test_alert_normalizes_code() -> None:
    item = alert(
        AlertSeverity.WARNING,
        " spread ",
        "wide",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert item.code == "SPREAD"


def test_fanout_publishes_to_all_sinks() -> None:
    first = InMemoryAlertSink()
    second = InMemoryAlertSink()
    item = alert(AlertSeverity.INFO, "READY", "ready")

    FanoutAlertSink((first, second)).emit(item)

    assert first.alerts == (item,)
    assert second.alerts == (item,)
