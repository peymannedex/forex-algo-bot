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

## Prerequisites

1. Keep MetaTrader 5 open and logged into a demo account.
2. Enable the configured symbols in Market Watch.
3. Append `config/.env.paper-live-feed.example` to the external `.env`.
4. Keep the real strategy selected for soak operation.

## Five-cycle commissioning test

```powershell
& ".\deploy\windows\run_paper_live_feed.ps1" `
  -RepoRoot "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo" `
  -EnvFile "C:\forex-algo-bot\config\.env" `
  -Strategy trend `
  -MaxCycles 5
```

This may take more than 25 minutes with M5 bars. A clean result ends with a JSON
summary and writes evidence under:

```text
C:\forex-algo-bot\evidence\paper-soak
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

The stop file is removed automatically at the next service start.

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
