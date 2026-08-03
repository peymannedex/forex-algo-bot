"""Production heartbeat, synchronization, and reconciliation supervisor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fxbot.execution.connection import MT5ConnectionManager, MT5ConnectionSnapshot
from fxbot.execution.reconciliation import LiveMT5Reconciler, MT5ReconciliationReport
from fxbot.execution.runtime import ExecutionRuntime, SyncResult
from fxbot.production.alerts import (
    AlertSeverity,
    AlertSink,
    LoggingAlertSink,
    alert,
)
from fxbot.production.checkpoint import (
    SupervisorCheckpoint,
    SupervisorCheckpointStore,
)
from fxbot.production.health import ComponentState, HealthRegistry

Clock = Callable[[], datetime]
ExpectedPositionsProvider = Callable[[], dict[str, float]]


@dataclass(frozen=True, slots=True)
class SupervisorCycleResult:
    checked_at: datetime
    connection: MT5ConnectionSnapshot
    sync: SyncResult
    reconciliation: MT5ReconciliationReport | None


class ProductionSupervisor:
    """Own connection health, runtime synchronization, and periodic reconciliation."""

    def __init__(
        self,
        *,
        connection: MT5ConnectionManager,
        runtime: ExecutionRuntime,
        reconciler: LiveMT5Reconciler,
        health: HealthRegistry,
        checkpoint_store: SupervisorCheckpointStore,
        expected_positions: ExpectedPositionsProvider,
        alert_sink: AlertSink | None = None,
        heartbeat_interval: timedelta | None = None,
        reconciliation_interval: timedelta | None = None,
        clock: Clock | None = None,
    ) -> None:
        resolved_heartbeat = heartbeat_interval or timedelta(seconds=5)
        resolved_reconciliation = reconciliation_interval or timedelta(seconds=30)
        if resolved_heartbeat <= timedelta(0):
            raise ValueError("heartbeat_interval must be positive")
        if resolved_reconciliation <= timedelta(0):
            raise ValueError("reconciliation_interval must be positive")
        self.connection = connection
        self.runtime = runtime
        self.reconciler = reconciler
        self.health = health
        self.checkpoint_store = checkpoint_store
        self.expected_positions = expected_positions
        self.alert_sink: AlertSink = alert_sink or LoggingAlertSink()
        self.heartbeat_interval = resolved_heartbeat
        self.reconciliation_interval = resolved_reconciliation
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._running = False
        self._checkpoint = self.checkpoint_store.load()

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> MT5ConnectionSnapshot:
        snapshot = self.connection.ensure_connected()
        self._running = True
        self.health.update(
            "connection",
            ComponentState.HEALTHY,
            "MT5 connection ready",
            metadata=(
                ("login", str(snapshot.account_login or "")),
                ("server", snapshot.server or ""),
            ),
            checked_at=snapshot.checked_at,
        )
        self.health.update(
            "execution",
            ComponentState.HEALTHY,
            "execution runtime ready",
        )
        return snapshot

    def run_once(self) -> SupervisorCycleResult:
        if not self._running:
            self.start()

        now = self._clock().astimezone(UTC)
        snapshot = self.connection.ensure_connected()
        connection_state = (
            ComponentState.HEALTHY
            if snapshot.connected and snapshot.trade_allowed
            else ComponentState.DEGRADED
        )
        self.health.update(
            "connection",
            connection_state,
            "MT5 heartbeat accepted"
            if connection_state is ComponentState.HEALTHY
            else "MT5 heartbeat degraded",
            checked_at=now,
        )

        sync_result = self.runtime.sync()
        execution_state = (
            ComponentState.DEGRADED if sync_result.warnings else ComponentState.HEALTHY
        )
        self.health.update(
            "execution",
            execution_state,
            (
                f"sync completed with {sync_result.warnings} warning(s)"
                if sync_result.warnings
                else "execution sync completed"
            ),
            checked_at=now,
            metadata=(
                ("new_fills", str(sync_result.new_fills)),
                ("order_updates", str(sync_result.order_updates)),
            ),
        )

        report: MT5ReconciliationReport | None = None
        last_reconcile = self._checkpoint.last_reconciliation_at
        due = (
            last_reconcile is None
            or now - last_reconcile >= self.reconciliation_interval
        )
        if due:
            since = last_reconcile or now - self.reconciliation_interval
            report = self.reconciler.reconcile(
                local_orders=self.runtime.known_orders,
                expected_positions=self.expected_positions(),
                since=since,
                checked_at=now,
            )
            state = ComponentState.HEALTHY if report.clean else ComponentState.DEGRADED
            self.health.update(
                "reconciliation",
                state,
                "broker state reconciled"
                if report.clean
                else f"reconciliation found {len(report.issues)} issue(s)",
                checked_at=now,
            )
            if not report.clean:
                self.alert_sink.emit(
                    alert(
                        AlertSeverity.WARNING,
                        "RECONCILIATION_ISSUES",
                        f"MT5 reconciliation found {len(report.issues)} issue(s)",
                        timestamp=now,
                    )
                )
            last_reconcile = now

        self._checkpoint = SupervisorCheckpoint(
            last_heartbeat_at=now,
            last_reconciliation_at=last_reconcile,
            last_seen_execution_id=self._checkpoint.last_seen_execution_id,
        )
        self.checkpoint_store.save(self._checkpoint)
        return SupervisorCycleResult(now, snapshot, sync_result, report)

    async def run(self, stop_event: asyncio.Event) -> None:
        self.start()
        try:
            while not stop_event.is_set():
                try:
                    self.run_once()
                except Exception as exc:
                    now = self._clock().astimezone(UTC)
                    self.health.update(
                        "supervisor",
                        ComponentState.UNHEALTHY,
                        f"{type(exc).__name__}: {exc}",
                        checked_at=now,
                    )
                    self.alert_sink.emit(
                        alert(
                            AlertSeverity.CRITICAL,
                            "SUPERVISOR_CYCLE_FAILED",
                            f"Production supervisor cycle failed: {exc}",
                            timestamp=now,
                        )
                    )
                    raise
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=self.heartbeat_interval.total_seconds(),
                    )
                except TimeoutError:
                    continue
        finally:
            self.stop()

    def stop(self) -> None:
        if self._running:
            self.connection.disconnect()
        self._running = False
        self.health.stop_all()
