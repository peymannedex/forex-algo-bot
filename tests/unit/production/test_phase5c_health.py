from datetime import UTC, datetime, timedelta

from fxbot.production.health import (
    ComponentState,
    HealthRegistry,
)


def test_ready_when_every_component_is_healthy() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry = HealthRegistry(clock=lambda: now)
    registry.update("connection", ComponentState.HEALTHY, "ok")
    registry.update("execution", ComponentState.HEALTHY, "ok")

    snapshot = registry.snapshot()

    assert snapshot.ready
    assert snapshot.state is ComponentState.HEALTHY


def test_unhealthy_dominates_snapshot() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry = HealthRegistry(clock=lambda: now)
    registry.update("connection", ComponentState.UNHEALTHY, "down")
    registry.update("execution", ComponentState.HEALTHY, "ok")

    assert registry.snapshot().state is ComponentState.UNHEALTHY


def test_stale_component_degrades() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    current = [base]
    registry = HealthRegistry(
        stale_after=timedelta(seconds=10),
        clock=lambda: current[0],
    )
    registry.update("connection", ComponentState.HEALTHY, "ok")
    current[0] = base + timedelta(seconds=11)

    snapshot = registry.snapshot()

    assert snapshot.state is ComponentState.DEGRADED
    assert "stale" in snapshot.components[0].message


def test_stop_all_marks_components_stopped() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry = HealthRegistry(clock=lambda: now)
    registry.update("connection", ComponentState.HEALTHY, "ok")

    registry.stop_all()

    assert registry.snapshot().state is ComponentState.STOPPED
