# Phase 5C Production Readiness Runbook

## Purpose

Phase 5C adds the operational boundary around the completed data, strategy,
risk, backtest, and MT5 execution layers. It does not enable real-money
trading automatically.

The production package provides:

- environment-based configuration with secret redaction;
- paper, demo, and live profiles;
- explicit confirmation gates for live trading;
- MT5 startup readiness checks;
- quote age and spread protections;
- UTC market-hours windows;
- daily-loss and peak-to-equity drawdown shutdown controls;
- shared execution kill-switch integration;
- emergency cancel and direct reduce-only flattening;
- connection heartbeat and execution synchronization;
- periodic order, fill, and position reconciliation;
- health, readiness, and liveness aggregation;
- structured JSON logs;
- operator alert protocols;
- atomic restart checkpoints;
- Windows preflight deployment scripts.

## Profile progression

### 1. Paper

Use:

```text
FXBOT_PROFILE=paper
```

Paper mode never enables broker submission. Run:

```powershell
python -m fxbot.production.bootstrap
```

The command validates configuration and exits without connecting to MT5.

### 2. Demo validation-only

Use:

```text
FXBOT_PROFILE=demo
FXBOT_DEMO_ORDER_SUBMISSION_ENABLED=false
```

The preflight connects to the demo terminal and validates account permissions,
but the MT5 adapter remains in `dry_run=True`.

### 3. Demo order submission

After validation-only checks and reconciliation are clean:

```text
FXBOT_PROFILE=demo
FXBOT_DEMO_ORDER_SUBMISSION_ENABLED=true
```

This allows actual orders only on the configured demo account. Verify the
server and account number in every readiness report.

### 4. Live

Live mode is rejected unless all gates are present:

```text
FXBOT_PROFILE=live
FXBOT_LIVE_TRADING_ENABLED=true
FXBOT_LIVE_CONFIRMATION=I_UNDERSTAND_LIVE_TRADING
FXBOT_MT5_LOGIN=...
FXBOT_MT5_PASSWORD=...
FXBOT_MT5_SERVER=...
```

Do not set these values until extended paper and demo testing are complete.

## Preflight

Copy the example environment file outside the repository, protect it with
Windows file permissions, and do not commit it.

```powershell
& ".\deploy\windows\run_preflight.ps1" `
  -RepoRoot "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo" `
  -EnvFile "C:\forex-algo-bot\config\.env"
```

A successful preflight returns exit code `0` and JSON containing
`"ready": true`.

## Supervisor integration

Build the Phase 5 execution components:

```python
from datetime import timedelta

from fxbot.production import (
    HealthRegistry,
    ProductionSettings,
    ProductionSupervisor,
    SupervisorCheckpointStore,
    build_mt5_components,
)

settings = ProductionSettings()
components = build_mt5_components(settings)

health = HealthRegistry(
    stale_after=timedelta(seconds=settings.health_stale_after_seconds)
)

supervisor = ProductionSupervisor(
    connection=components.connection,
    runtime=components.runtime,
    reconciler=components.reconciler,
    health=health,
    checkpoint_store=SupervisorCheckpointStore(
        settings.state_directory / "supervisor.json"
    ),
    expected_positions=lambda: {},
    heartbeat_interval=timedelta(
        seconds=settings.heartbeat_interval_seconds
    ),
    reconciliation_interval=timedelta(
        seconds=settings.reconciliation_interval_seconds
    ),
)
```

Replace `expected_positions=lambda: {}` with the signed position state owned by
the portfolio/risk layer before demo order submission.

## Restart-safe fill processing

Wrap position, accounting, or portfolio fill consumers with
`RecoverableFillSink` and a `FileExecutionJournal` stored outside the Git
checkout:

```python
from fxbot.production import FileExecutionJournal, RecoverableFillSink

journal = FileExecutionJournal(
    settings.state_directory / "execution-journal.json"
)
durable_sink = RecoverableFillSink(journal, downstream_fill_sink)
components = build_mt5_components(
    settings,
    fill_sinks=(durable_sink,),
)
```

A fill is marked pending before downstream processing and committed only after
the downstream sink succeeds. Pending IDs are retried after restart; committed
IDs are suppressed.

## Emergency workflow

The emergency controller disables ordinary routing first. It then cancels
open orders and, when explicitly requested, sends reduce-only market orders
directly through the broker adapter so the shared kill switch cannot block
risk-reducing liquidation.

Automatic flattening remains disabled by default:

```text
FXBOT_AUTO_FLATTEN_ON_TRIP=false
```

Test emergency cancellation and flattening repeatedly on a demo account.

## Logging

`configure_json_logging()` writes one JSON object per line. Store logs outside
the Git checkout and apply retention and access controls. Never include MT5
passwords or live confirmation values in log context.

## Acceptance gates before live use

All of these must be documented:

1. At least four continuous weeks of paper operation without duplicate orders.
2. At least two weeks of demo order submission with daily reconciliation.
3. Successful restart recovery while positions and pending orders exist.
4. Successful simulated network interruption and MT5 terminal restart.
5. Successful stale-quote, spread, daily-loss, and drawdown trips.
6. Successful emergency cancellation and flattening on the demo account.
7. Verified account number, server, symbol suffixes, volume steps, and stop levels.
8. External review of strategy, risk, deployment, and secret-management settings.
9. Backups of configuration and operational state.
10. A human operator available whenever live execution is enabled.

## Important limitation

Phase 5C supplies the production primitives and preflight. The repository's
strategy application must still wire its signal loop, expected signed
positions, account-equity snapshots, quote guards, and shutdown policy into
these components. No package can make an unvalidated strategy safe for
real-money trading.
