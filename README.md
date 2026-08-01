# Modular Algorithmic Forex Trading Bot

A production-oriented, event-driven platform for Forex market-data ingestion,
strategy research, portfolio risk control, realistic backtesting, and broker
execution. The project keeps data, strategy, risk, execution, and observability
strictly separated so each layer can be tested and replaced independently.

> **Current status:** Phase 1B — canonical market-data models, historical CSV/Parquet
> and MT5 adapters, exact tick-to-bar bid/ask/mid resampling, atomic monthly
> Parquet partition storage, asynchronous live-data boundaries, and conservative
> data-quality controls.

## Design principles

- **Executable-price fidelity:** ticks and bars preserve both bid and ask; the
  backtester will never assume a mid-price is directly tradable.
- **UTC internally:** all source timestamps are normalized to timezone-aware UTC.
- **Immutable domain models:** downstream modules cannot accidentally mutate past
  market events.
- **Streaming first:** CSV and Parquet ingestion avoid loading entire tick archives
  into memory.
- **Event driven:** live adapters emit canonical market-data records to a future
  event bus, not directly to strategies.
- **Conservative cleaning:** deterministic defects can be removed; statistical
  jumps are flagged by default rather than silently deleted.
- **Broker isolation:** MT5, cTrader, REST, WebSocket, or FIX code will live behind
  execution and data-adapter interfaces.
- **Reproducibility:** configuration, tests, experiment metadata, and CI are part
  of the repository from the first commit.

## Repository layout

```text
.
├── .github/workflows/       # Continuous integration
├── config/                  # Versioned non-secret configuration
├── data/                    # Local raw/interim/processed data; contents ignored
├── docs/                    # Architecture decisions and research notes
├── mql5/                    # Future MT5 execution bridge and Expert Advisor
├── scripts/                 # Operational and data-management entry points
├── src/fxbot/
│   ├── domain/              # Canonical immutable business objects
│   ├── data/                # Ingestion, cleaning, resampling, storage, adapters
│   ├── events/              # Event bus and event contracts (Phase 4)
│   ├── features/            # Indicators and regime features (Phase 3)
│   ├── strategies/          # Signal engines (Phase 3)
│   ├── risk/                # Position sizing and hard controls (Phase 2)
│   ├── portfolio/           # Currency exposure and allocation (Phase 2)
│   ├── backtest/            # Event-driven simulator (Phase 4)
│   ├── execution/           # Broker gateways and OMS (Phase 5)
│   └── monitoring/          # Logging, metrics, alerts, kill switches
└── tests/
    ├── unit/
    └── integration/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Run validation:

```bash
ruff check src tests
mypy src
pytest --cov=fxbot --cov-report=term-missing
```

## Phase 1 data-engine usage

### Historical tick CSV

Expected source columns can be renamed through `TickCSVSchema`.

```python
from datetime import UTC, datetime

from fxbot.data.adapters import CSVMarketDataAdapter
from fxbot.data.cleaning import DataCleaningConfig, MarketDataCleaner
from fxbot.data.schemas import TickCSVSchema
from fxbot.domain import DataKind, HistoricalDataRequest, SymbolSpec

spec = SymbolSpec(
    symbol="EURUSD",
    base_currency="EUR",
    quote_currency="USD",
    digits=5,
    point_size=0.00001,
    pip_size=0.0001,
)

adapter = CSVMarketDataAdapter(
    "data/raw/EURUSD_ticks_*.csv",
    tick_schema=TickCSVSchema(
        timestamp="timestamp",
        symbol="symbol",
        bid="bid",
        ask="ask",
        bid_size=None,
        ask_size=None,
        sequence=None,
    ),
)

request = HistoricalDataRequest(
    symbol="EURUSD",
    kind=DataKind.TICK,
    start=datetime(2025, 1, 1, tzinfo=UTC),
    end=datetime(2026, 1, 1, tzinfo=UTC),
)

raw_ticks = adapter.iter_ticks(request)
cleaned = MarketDataCleaner(
    DataCleaningConfig(
        sort_records=True,
        remove_exact_duplicates=True,
        max_spread_pips=5.0,
        detect_return_outliers=True,
        drop_return_outliers=False,
    ),
    {"EURUSD": spec},
).clean_ticks(raw_ticks)

print(cleaned.report)
```


### Tick-to-bar resampling

```python
from fxbot.data import TickBarResampler
from fxbot.domain import Timeframe

resampler = TickBarResampler(Timeframe.M1)
for completed_bar in resampler.resample(cleaned.records):
    print(completed_bar.bid, completed_bar.ask, completed_bar.mid)
```

The resampler aggregates true tick-level mid prices. It does not infer a mid
high or low by averaging independently timed bid and ask extrema, and it does
not fabricate empty bars for missing buckets. Weekly bars align to Monday
00:00 UTC.

### Atomic Parquet partition store

```python
from fxbot.data import ParquetPartitionStore

store = ParquetPartitionStore("data/processed/market")
write_result = store.append_ticks(cleaned.records)
print(write_result.files_created)

restored = store.iter_ticks(request)
```

The store appends immutable part files under paths such as
`kind=tick/symbol=EURUSD/year=2026/month=01/`. Bar partitions also include a
`timeframe=...` component. Writes use a temporary file followed by an atomic
rename, and stored files carry an explicit schema version.

### MetaTrader 5 data adapter

The official MetaTrader5 Python bridge is installed only on Windows hosts. The
terminal must be installed and available to the Python process.

```python
import asyncio

from fxbot.data.adapters import MT5MarketDataAdapter
from fxbot.domain import LiveSubscription, Timeframe


async def stream_mt5() -> None:
    adapter = MT5MarketDataAdapter(poll_interval_seconds=0.10)
    subscription = LiveSubscription(
        symbols=frozenset({"EURUSD"}),
        timeframes=frozenset({Timeframe.TICK, Timeframe.M1}),
    )
    async with adapter:
        async for record in adapter.stream(subscription):
            print(record)


asyncio.run(stream_mt5())
```

MT5 native rate data contain bid OHLC plus spread in points. The adapter
reconstructs ask OHLC with that bar-level spread and labels the source
`mt5:bid-plus-spread`. For exact bid/ask/mid candles, extract ticks and use the
resampler. The MT5 Python API is synchronous, so live polling is delegated to a
worker thread rather than blocking the event loop.

### Historical OHLC bars

Native bid/ask OHLC is the preferred schema:

```text
timestamp,symbol,timeframe,bid_open,bid_high,bid_low,bid_close,
ask_open,ask_high,ask_low,ask_close,tick_volume
```

`BarCSVSchema` also supports `bid_plus_spread` and `mid_plus_spread` source
formats. Those modes are approximations and should be recorded in backtest
metadata.

### Live market-data boundary

```python
import asyncio
from datetime import UTC, datetime

from fxbot.data.adapters import AsyncQueueLiveDataAdapter
from fxbot.domain import LiveSubscription, Tick, Timeframe


async def main() -> None:
    adapter = AsyncQueueLiveDataAdapter(max_queue_size=100_000)
    subscription = LiveSubscription(
        symbols=frozenset({"EURUSD"}),
        timeframes=frozenset({Timeframe.TICK}),
    )

    async with adapter:
        await adapter.publish(
            Tick(
                symbol="EURUSD",
                event_time=datetime.now(UTC),
                bid=1.10000,
                ask=1.10012,
                source="demo-feed",
            )
        )
        stream = adapter.stream(subscription)
        record = await anext(stream)
        print(record)
        await stream.aclose()


asyncio.run(main())
```

A broker connector will later transform native messages into these immutable
records and call `publish`. Strategies will never depend on broker payloads.

## Data invariants

- `ask >= bid`; crossed quotes are invalid.
- prices are positive finite numbers.
- timestamps are timezone-aware and stored in UTC.
- historical ranges are half-open: `start <= timestamp < end`.
- bars cannot use the tick timeframe.
- bid and ask OHLC envelopes must each be internally valid.
- spread-derived bars are allowed only through an explicit schema mode.
- exact duplicate removal is auditable through `CleaningReport`.
- return outlier deletion is opt-in and disabled by default.

## Five development milestones

### Phase 1 — Data engine and canonical models

- Tick, bid/ask OHLC, symbol, timeframe, and query models
- CSV/Parquet historical adapters
- Live adapter interfaces and queue-based integration boundary
- Data-quality reports, duplicate handling, gap detection, spread checks
- Exact bid/ask/mid tick resampling and UTC bucket alignment
- Atomic Parquet stores partitioned by symbol, year, and month
- MT5 historical extraction and asynchronous polling adapter
- Next: data catalogs, compaction, retention, and feed-health telemetry

### Phase 2 — Risk and portfolio manager

- Fixed-fractional and volatility-targeted position sizing
- Pip value and account-currency conversion
- Currency-level exposure decomposition
- Correlation and concentration limits
- Daily loss, equity drawdown, margin, and maximum-trade controls

### Phase 3 — Feature, regime, and strategy engines

- ATR, realized volatility, directional efficiency, structure, sessions
- Multi-timeframe alignment using completed bars only
- Trend, breakout, mean-reversion, and quantified liquidity-sweep baselines
- Deterministic signal lifecycle and strategy state machines

### Phase 4 — Event-driven backtester and validation

- Shared event contracts for historical replay and live operation
- Bid/ask execution, variable spread, commission, swaps, latency, and slippage
- Order/position accounting and portfolio attribution
- Walk-forward analysis, Monte Carlo stress tests, and overfitting diagnostics

### Phase 5 — Live execution and reconciliation

- MT5/MQL5 or cTrader execution gateway (market-data MT5 adapter exists in Phase 1B)
- Idempotent order management and broker-side protective stops
- Position/order reconciliation with broker state as source of truth
- Economic-calendar controls, monitoring, alerts, and independent kill switches
- Shadow, paper, minimum-risk, and controlled scaling deployment stages

## Security and operational rules

- Never commit broker credentials, tokens, account IDs, or private keys.
- Broker-side protective stops are mandatory before unattended live trading.
- The live system must stop on stale data, state mismatch, abnormal spread,
  excessive slippage, or breached equity limits.
- Backtest and live code must share domain and order-state definitions to reduce
  simulation/live divergence.

## License

Proprietary until a project license is selected.
