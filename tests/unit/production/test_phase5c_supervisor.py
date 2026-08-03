import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from fxbot.execution.connection import MT5ConnectionSnapshot
from fxbot.execution.reconciliation import MT5ReconciliationReport
from fxbot.execution.runtime import SyncResult
from fxbot.production.alerts import InMemoryAlertSink
from fxbot.production.checkpoint import SupervisorCheckpointStore
from fxbot.production.health import ComponentState, HealthRegistry
from fxbot.production.supervisor import ProductionSupervisor


class Connection:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.disconnected = False

    def ensure_connected(self) -> MT5ConnectionSnapshot:
        return MT5ConnectionSnapshot(True, 123, "Demo", True, self.now)

    def disconnect(self) -> None:
        self.disconnected = True


class Runtime:
    known_orders = ()

    def sync(self) -> SyncResult:
        return SyncResult(1, 0, 2, 0)


class Reconciler:
    def __init__(self, clean: bool = True) -> None:
        self.clean = clean
        self.calls = 0

    def reconcile(self, **kwargs) -> MT5ReconciliationReport:
        self.calls += 1
        issues = () if self.clean else (SimpleNamespace(),)
        return MT5ReconciliationReport(
            checked_at=kwargs["checked_at"],
            broker_orders=(),
            broker_positions=(),
            recovered_fills=(),
            issues=issues,
        )


def make_supervisor(tmp_path, *, clean=True):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    current = [now]
    connection = Connection(now)
    health = HealthRegistry(clock=lambda: current[0])
    alerts = InMemoryAlertSink()
    reconciler = Reconciler(clean=clean)
    supervisor = ProductionSupervisor(
        connection=connection,
        runtime=Runtime(),
        reconciler=reconciler,
        health=health,
        checkpoint_store=SupervisorCheckpointStore(tmp_path / "state.json"),
        expected_positions=lambda: {},
        alert_sink=alerts,
        heartbeat_interval=timedelta(milliseconds=1),
        reconciliation_interval=timedelta(seconds=30),
        clock=lambda: current[0],
    )
    return supervisor, connection, health, alerts, reconciler, current


def test_run_once_updates_health_and_checkpoint(tmp_path) -> None:
    supervisor, _, health, _, reconciler, _ = make_supervisor(tmp_path)

    result = supervisor.run_once()

    assert result.sync.new_fills == 1
    assert reconciler.calls == 1
    assert health.snapshot().state is ComponentState.HEALTHY
    assert (tmp_path / "state.json").exists()


def test_dirty_reconciliation_emits_alert(tmp_path) -> None:
    supervisor, _, health, alerts, _, _ = make_supervisor(
        tmp_path,
        clean=False,
    )

    result = supervisor.run_once()

    assert result.reconciliation is not None
    assert not result.reconciliation.clean
    assert alerts.alerts
    assert health.snapshot().state is ComponentState.DEGRADED


@pytest.mark.asyncio
async def test_async_run_stops_gracefully(tmp_path) -> None:
    supervisor, connection, health, _, _, _ = make_supervisor(tmp_path)
    stop = asyncio.Event()

    async def set_stop() -> None:
        await asyncio.sleep(0.005)
        stop.set()

    await asyncio.gather(supervisor.run(stop), set_stop())

    assert connection.disconnected
    assert health.snapshot().state is ComponentState.STOPPED
