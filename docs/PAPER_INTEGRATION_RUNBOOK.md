# Paper Integration and Acceptance Runbook

This stage joins the completed data, strategy, risk, execution, and production
layers without connecting to a live broker. It is the first end-to-end runtime
that can produce strategy decisions, size orders, enforce safeguards, execute
through the deterministic paper broker, consume fills exactly once, update a
portfolio ledger, persist restart state, and report component health.

## Safety boundary

Keep the external environment file on the paper profile:

```text
FXBOT_PROFILE=paper
FXBOT_LIVE_TRADING_ENABLED=false
FXBOT_DEMO_ORDER_SUBMISSION_ENABLED=false
```

The paper integration runtime never constructs the MT5 adapter. It uses
`PaperBroker` exclusively.

## Configuration

Append the values from `config/.env.paper-integration.example` to:

```text
C:\forex-algo-bot\config\.env
```

For the default EURUSD acceptance run, risk-based Phase 2 sizing can be used
because the quote currency and account currency are both USD. Set
`FXBOT_PAPER_FIXED_QUANTITY=0.01` for a fixed-lot smoke test.

## One-command deterministic acceptance replay

```powershell
Set-ExecutionPolicy -Scope Process Bypass

& ".\deploy\windows\run_paper_acceptance.ps1" `
  -RepoRoot "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo" `
  -EnvFile "C:\forex-algo-bot\config\.env" `
  -ReplayCsv "C:\forex-algo-bot\data\paper_replay.csv"
```

The script generates chronological M5 and M15 bid/ask bars, clears prior paper
state unless `-KeepState` is supplied, runs the deterministic acceptance smoke strategy through
the complete paper stack, and prints one JSON result. The smoke strategy is not a
trading strategy; it exists only to force entry, exit, reversal, and fill paths.

A successful result contains:

```json
{
  "profile": "paper",
  "ready": true,
  "summary": {
    "cycles": 360
  }
}
```

The acceptance smoke replay should produce orders and fills. A separate run with
`--strategy trend` exercises the real trend-following strategy; its exact trade
count is data- and threshold-dependent. Readiness, clean
completion, deterministic repeatability, and durable state are the acceptance
criteria; profitability on synthetic data is not.

## Runtime sequence

Each primary-timeframe bar close executes this sequence:

1. Build a look-ahead-safe multi-timeframe context.
2. Publish the executable bid/ask quote to the paper broker and ledger.
3. Consume fills and order updates from prior pending orders.
4. Run the strategy readiness and duplicate-signal safeguards.
5. Convert the strategy decision into net-position order intents.
6. Size entries through the Phase 2 position sizer or fixed-lot policy.
7. Apply quote age, spread, market hours, account loss, drawdown, and exposure limits.
8. Route orders idempotently through `ExecutionRouter`.
9. Consume immediate paper fills exactly once through the execution journal.
10. Mark balance, equity, positions, health, and the restart checkpoint.

## Restart test

Run once with state reset, then run with `-KeepState` and a replay containing
only timestamps after the saved checkpoint. The runtime rejects duplicate or
out-of-order frames by design.

State files are stored under the configured `FXBOT_STATE_DIRECTORY`:

```text
paper_runtime_state.json
paper_execution_journal.json
```

## Required acceptance evidence

Before moving to MT5 demo order submission, retain evidence of:

- Four continuous weeks of paper-mode operation.
- No unhandled runtime exceptions.
- No duplicate execution IDs.
- Daily health snapshots remaining ready or explicitly explained.
- Deterministic replay results from the same input and clean initial state.
- Verified restart recovery from saved ledger and execution-journal state.
- Deliberate tests of stale quote, wide spread, market-closed, loss-limit, and exposure-limit rejection.

After paper acceptance, use an MT5 demo account for at least two weeks with
daily order, fill, and position reconciliation. Real-money activation remains a
separate operational decision and is not performed by this package.
