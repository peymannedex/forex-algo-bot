# Paper Live-Feed Soak Runbook

This service uses the existing MT5 **market-data adapter only** and routes every
strategy order to the existing deterministic `PaperBroker`. It never constructs
the MT5 execution adapter.

## Mandatory safety settings

```text
FXBOT_PROFILE=paper
FXBOT_LIVE_TRADING_ENABLED=false
FXBOT_DEMO_ORDER_SUBMISSION_ENABLED=false
```

## MT5 server-time normalization

Some MT5 brokers expose server-local timestamps through the Python bridge.
Measure the difference between a current MT5 tick and real UTC, then configure
the positive number of minutes by which MT5 is ahead of UTC.

For a terminal returning timestamps three hours ahead:

```text
FXBOT_PAPER_LIVE_MT5_SERVER_UTC_OFFSET_MINUTES=180
FXBOT_PAPER_LIVE_MAX_FUTURE_SKEW_SECONDS=5
```

The live source subtracts the configured offset before records enter the
strategy. Any record that remains future-dated beyond the configured tolerance
is rejected. Recheck this value when the broker changes daylight-saving time.

## Five-cycle commissioning test

```powershell
& ".\deploy\windows\run_paper_live_feed.ps1" `
  -RepoRoot "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo" `
  -EnvFile "C:\forex-algo-bot\config\.env" `
  -Strategy trend `
  -MaxCycles 5
```

## Continuous soak

```powershell
& ".\deploy\windows\run_paper_live_feed.ps1" `
  -RepoRoot "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo" `
  -EnvFile "C:\forex-algo-bot\config\.env" `
  -Strategy trend
```

Stop with `Ctrl+C`, or create:

```text
C:\forex-algo-bot\var\state\STOP_PAPER_SOAK
```

## Evidence

- `cycles.jsonl`: one record per processed primary bar.
- `errors.jsonl`: reconnect and processing errors.
- `latest-summary.json`: atomic current process summary.
- `daily-YYYY-MM-DD.json`: latest summary for each UTC day.
- Existing `paper_runtime_state.json`: ledger and cycle checkpoint.
- Existing `paper_execution_journal.json`: committed fill IDs.

## Acceptance boundary

Run continuously for four weeks. Review the daily files and explain every
degraded or unhealthy interval. Do not move to MT5 demo-order submission until
the paper-soak requirement is complete.
