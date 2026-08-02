from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fxbot.data.adapters.base import LiveMarketDataAdapter
from fxbot.data.catalog import DataCatalog
from fxbot.data.checkpoints import CheckpointKey, CheckpointStore
from fxbot.data.cleaning import MarketDataCleaner
from fxbot.data.live_pipeline import (
    GapRecoveryPolicy,
    LiveBatchDisposition,
    LiveBatchPersister,
    LiveCheckpointError,
    LiveGapRecoveryError,
    LiveIngestionConfig,
    LiveIngestionConfigurationError,
    LiveIngestionQualityError,
    LiveIngestionService,
    LivePersistenceError,
    LiveReconnectExhaustedError,
    MicroBatchPolicy,
)
from fxbot.data.pipeline import (
    HistoricalIngestionPlan,
    HistoricalIngestionResult,
    QualityRejectionPolicy,
)
from fxbot.data.quality import DataQualityGate, QualityThresholds
from fxbot.data.quarantine import JsonQuarantineStore
from fxbot.data.retry import ReconnectPolicy
from fxbot.data.storage import PartitionRef, StorageWriteResult
from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import OHLC, Bar, LiveSubscription, MarketDataRecord, Tick

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def tick(second: int, *, symbol: str = "EURUSD", sequence: int | None = None) -> Tick:
    return Tick(
        symbol=symbol,
        event_time=BASE + timedelta(seconds=second),
        bid=1.1000 + second * 0.000001,
        ask=1.1002 + second * 0.000001,
        source="fake",
        sequence=sequence,
    )


def bar(minute: int, *, complete: bool = True, symbol: str = "EURUSD") -> Bar:
    value = 1.10 + minute * 0.001
    return Bar(
        symbol=symbol,
        open_time=BASE + timedelta(minutes=minute),
        timeframe=Timeframe.M1,
        bid=OHLC(value, value + 0.001, value - 0.001, value + 0.0005),
        ask=OHLC(value + 0.0002, value + 0.0012, value - 0.0008, value + 0.0007),
        source="fake",
        complete=complete,
    )


class FileStorage:
    def __init__(
        self,
        root: Path,
        *,
        reported_count: int | None = None,
        call_event: threading.Event | None = None,
    ) -> None:
        self.root = root
        self.reported_count = reported_count
        self.call_event = call_event
        self.calls = 0

    def append(self, records: Iterable[MarketDataRecord]) -> StorageWriteResult:
        items = tuple(records)
        self.calls += 1
        path = self.root / f"part-{self.calls}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"live-test")
        first = items[0]
        partition = (
            PartitionRef(DataKind.TICK, first.symbol, 2026, 1)
            if isinstance(first, Tick)
            else PartitionRef(DataKind.BAR, first.symbol, 2026, 1, first.timeframe)
        )
        if self.call_event is not None:
            self.call_event.set()
        return StorageWriteResult(
            records_written=(
                self.reported_count if self.reported_count is not None else len(items)
            ),
            files_created=(path,),
            partitions=(partition,),
        )


SessionFactory = Callable[[], AsyncIterator[MarketDataRecord]]


class FakeLiveAdapter(LiveMarketDataAdapter):
    def __init__(
        self,
        sessions: list[SessionFactory],
        *,
        connect_errors: list[Exception | None] | None = None,
    ) -> None:
        self.sessions = sessions
        self.connect_errors = connect_errors or []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self._connected = False
        self._session_index = -1

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        index = self.connect_calls
        self.connect_calls += 1
        if index < len(self.connect_errors) and self.connect_errors[index] is not None:
            raise self.connect_errors[index]  # type: ignore[misc]
        self._connected = True
        self._session_index += 1

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    def stream(self, subscription: LiveSubscription) -> AsyncIterator[MarketDataRecord]:
        del subscription
        return self.sessions[self._session_index]()


class FailingCheckpointStore:
    def get(self, key: CheckpointKey) -> None:
        del key
        return None

    def advance(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("checkpoint unavailable")


class FakeRecovery:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.plans: list[HistoricalIngestionPlan] = []
        self.error = error

    def run(self, plan: HistoricalIngestionPlan) -> HistoricalIngestionResult:
        self.plans.append(plan)
        if self.error is not None:
            raise self.error
        return HistoricalIngestionResult(plan=plan, chunks=())


def session(*items: MarketDataRecord | Exception) -> SessionFactory:
    async def generate() -> AsyncIterator[MarketDataRecord]:
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item

    return generate


def config(
    *,
    max_records: int = 10,
    max_interval: timedelta = timedelta(seconds=1),
    reconnect: ReconnectPolicy | None = None,
    gap_recovery: GapRecoveryPolicy | None = None,
    quality_policy: QualityRejectionPolicy = QualityRejectionPolicy.RAISE,
    persist_incomplete_bars: bool = False,
    restart_on_stream_end: bool = False,
) -> LiveIngestionConfig:
    return LiveIngestionConfig(
        pipeline_id="live-test",
        source="fake",
        micro_batch=MicroBatchPolicy(
            max_records=max_records,
            max_interval=max_interval,
            queue_maxsize=100,
            stop_poll_interval_seconds=0.005,
        ),
        reconnect=reconnect or ReconnectPolicy(max_attempts=0, jitter_ratio=0),
        gap_recovery=gap_recovery or GapRecoveryPolicy(enabled=False),
        quality_rejection_policy=quality_policy,
        persist_incomplete_bars=persist_incomplete_bars,
        restart_on_stream_end=restart_on_stream_end,
    )


def persister(
    tmp_path: Path,
    cfg: LiveIngestionConfig,
    *,
    storage: FileStorage | None = None,
    checkpoints: Any | None = None,
    gate: DataQualityGate | None = None,
    quarantine: bool = False,
) -> tuple[LiveBatchPersister, FileStorage, DataCatalog, Any]:
    actual_storage = storage or FileStorage(tmp_path / "parts")
    catalog = DataCatalog(tmp_path / "state.sqlite3")
    actual_checkpoints = checkpoints or CheckpointStore(tmp_path / "state.sqlite3")
    return (
        LiveBatchPersister(
            config=cfg,
            cleaner=MarketDataCleaner(),
            quality_gate=gate or DataQualityGate(),
            storage=actual_storage,
            catalog=catalog,
            checkpoints=actual_checkpoints,
            quarantine=(
                JsonQuarantineStore(tmp_path / "quarantine") if quarantine else None
            ),
        ),
        actual_storage,
        catalog,
        actual_checkpoints,
    )


def subscription(*timeframes: Timeframe, symbols: frozenset[str] | None = None) -> LiveSubscription:
    return LiveSubscription(
        symbols=symbols or frozenset({"EURUSD"}),
        timeframes=frozenset(timeframes or (Timeframe.TICK,)),
    )


def test_policy_models_validate_values() -> None:
    with pytest.raises(ValueError, match="max_records"):
        MicroBatchPolicy(max_records=0)
    with pytest.raises(ValueError, match="max_interval"):
        MicroBatchPolicy(max_interval=timedelta(0))
    with pytest.raises(ValueError, match="queue_maxsize"):
        MicroBatchPolicy(queue_maxsize=0)
    with pytest.raises(ValueError, match="stop_poll"):
        MicroBatchPolicy(stop_poll_interval_seconds=0)
    with pytest.raises(ValueError, match="safety_lag"):
        GapRecoveryPolicy(safety_lag=timedelta(seconds=-1))
    with pytest.raises(ValueError, match="min_gap"):
        GapRecoveryPolicy(min_gap=timedelta(seconds=-1))
    with pytest.raises(ValueError, match="chunk_size"):
        GapRecoveryPolicy(chunk_size=timedelta(0))
    with pytest.raises(ValueError, match="initial_lookback"):
        GapRecoveryPolicy(initial_lookback=timedelta(0))
    with pytest.raises(ValueError, match="pipeline_id"):
        LiveIngestionConfig("", "fake")
    with pytest.raises(ValueError, match="source"):
        LiveIngestionConfig("live", " ")
    with pytest.raises(ValueError, match="schema_version"):
        LiveIngestionConfig("live", "fake", schema_version=0)
    with pytest.raises(ValueError, match="quarantine_sample_size"):
        LiveIngestionConfig("live", "fake", quarantine_sample_size=-1)


def test_persister_writes_reuses_and_advances_checkpoint(tmp_path: Path) -> None:
    cfg = config()
    target, storage, _, checkpoints = persister(tmp_path, cfg)
    records = (tick(0, sequence=1), tick(1, sequence=2))

    first = target.persist(records)
    second = target.persist(records)

    assert first.disposition is LiveBatchDisposition.WRITTEN
    assert first.batch is not None
    assert first.checkpoint is not None
    assert second.disposition is LiveBatchDisposition.ALREADY_REGISTERED
    assert second.batch == first.batch
    assert storage.calls == 1
    saved = checkpoints.get(CheckpointKey("live-test", "fake", "EURUSD", DataKind.TICK))
    assert saved is not None
    assert saved.last_event_time == BASE + timedelta(seconds=1)
    assert saved.last_sequence == 2


def test_persister_rejects_mixed_streams(tmp_path: Path) -> None:
    target, _, _, _ = persister(tmp_path, config())
    with pytest.raises(ValueError, match="one data stream"):
        target.persist((tick(0), tick(1, symbol="GBPUSD")))


def test_persister_quarantines_or_raises_quality_failure(tmp_path: Path) -> None:
    gate = DataQualityGate(QualityThresholds(min_output_records=3))
    continue_cfg = config(quality_policy=QualityRejectionPolicy.QUARANTINE_CONTINUE)
    target, storage, _, _ = persister(
        tmp_path,
        continue_cfg,
        gate=gate,
        quarantine=True,
    )
    result = target.persist((tick(0), tick(1)))
    assert result.disposition is LiveBatchDisposition.QUARANTINED
    assert result.quarantine_file is not None and result.quarantine_file.exists()
    assert storage.calls == 0

    raise_cfg = config(quality_policy=QualityRejectionPolicy.RAISE)
    raising, _, _, _ = persister(
        tmp_path / "raise",
        raise_cfg,
        gate=gate,
        quarantine=True,
    )
    with pytest.raises(LiveIngestionQualityError) as captured:
        raising.persist((tick(0), tick(1)))
    assert captured.value.quarantine_file is not None


def test_persister_cleans_failed_storage_result(tmp_path: Path) -> None:
    bad_storage = FileStorage(tmp_path / "parts", reported_count=99)
    target, _, _, _ = persister(tmp_path, config(), storage=bad_storage)

    with pytest.raises(LivePersistenceError, match="reported"):
        target.persist((tick(0),))
    assert not tuple((tmp_path / "parts").glob("*.parquet"))


def test_persister_reports_durable_checkpoint_failure(tmp_path: Path) -> None:
    target, storage, _, _ = persister(
        tmp_path,
        config(),
        checkpoints=FailingCheckpointStore(),
    )
    with pytest.raises(LiveCheckpointError) as captured:
        target.persist((tick(0),))
    assert captured.value.batch.files
    assert storage.calls == 1


@pytest.mark.asyncio
async def test_service_flushes_at_record_limit(tmp_path: Path) -> None:
    cfg = config(max_records=2)
    target, storage, _, _ = persister(tmp_path, cfg)
    adapter = FakeLiveAdapter([session(tick(0), tick(1))])
    service = LiveIngestionService(adapter=adapter, persister=target, config=cfg)

    result = await service.run(subscription(Timeframe.TICK))

    assert result.received_records == 2
    assert result.accepted_records == 2
    assert result.written_batches == 1
    assert storage.calls == 1
    assert adapter.connect_calls == 1
    assert adapter.disconnect_calls >= 1


@pytest.mark.asyncio
async def test_service_flushes_due_batch_while_stream_is_idle(tmp_path: Path) -> None:
    persisted = threading.Event()

    async def idle_session() -> AsyncIterator[MarketDataRecord]:
        yield tick(0)
        completed = await asyncio.to_thread(persisted.wait, 1.0)
        assert completed

    cfg = config(max_interval=timedelta(milliseconds=10))
    target, storage, _, _ = persister(
        tmp_path,
        cfg,
        storage=FileStorage(tmp_path / "parts", call_event=persisted),
    )
    service = LiveIngestionService(
        adapter=FakeLiveAdapter([idle_session]),
        persister=target,
        config=cfg,
    )

    result = await service.run(subscription(Timeframe.TICK))
    assert result.written_batches == 1
    assert storage.calls == 1


@pytest.mark.asyncio
async def test_service_separates_streams_and_drops_incomplete_bars(tmp_path: Path) -> None:
    cfg = config(max_records=10)
    target, storage, _, _ = persister(tmp_path, cfg)
    adapter = FakeLiveAdapter(
        [session(tick(0), tick(1, symbol="GBPUSD"), bar(0, complete=False), bar(1))]
    )
    service = LiveIngestionService(adapter=adapter, persister=target, config=cfg)

    result = await service.run(
        subscription(
            Timeframe.TICK,
            Timeframe.M1,
            symbols=frozenset({"EURUSD", "GBPUSD"}),
        )
    )

    assert result.received_records == 4
    assert result.accepted_records == 3
    assert result.dropped_incomplete_bars == 1
    assert result.written_batches == 3
    assert storage.calls == 3


@pytest.mark.asyncio
async def test_service_reconnects_after_transient_stream_error(tmp_path: Path) -> None:
    cfg = config(
        reconnect=ReconnectPolicy(
            initial_delay_seconds=0.01,
            max_delay_seconds=0.01,
            jitter_ratio=0,
            max_attempts=2,
        ),
        restart_on_stream_end=False,
    )
    target, _, _, _ = persister(tmp_path, cfg)
    adapter = FakeLiveAdapter([session(OSError("socket reset")), session(tick(0))])
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    service = LiveIngestionService(
        adapter=adapter,
        persister=target,
        config=cfg,
        sleep=fake_sleep,
        random_value=lambda: 0.5,
    )
    result = await service.run(subscription(Timeframe.TICK))

    assert result.reconnects == 1
    assert result.transient_errors == ("OSError: socket reset",)
    assert delays == [0.01]
    assert result.written_batches == 1
    assert adapter.connect_calls == 2


@pytest.mark.asyncio
async def test_service_raises_when_reconnects_are_exhausted(tmp_path: Path) -> None:
    cfg = config(
        reconnect=ReconnectPolicy(
            initial_delay_seconds=0.001,
            max_delay_seconds=0.001,
            jitter_ratio=0,
            max_attempts=1,
        )
    )
    target, _, _, _ = persister(tmp_path, cfg)
    adapter = FakeLiveAdapter(
        [session()],
        connect_errors=[OSError("offline"), OSError("still offline")],
    )
    service = LiveIngestionService(adapter=adapter, persister=target, config=cfg)

    with pytest.raises(LiveReconnectExhaustedError) as captured:
        await service.run(subscription(Timeframe.TICK))
    assert captured.value.attempts == 1
    assert adapter.connect_calls == 2


@pytest.mark.asyncio
async def test_service_request_stop_flushes_buffer(tmp_path: Path) -> None:
    yielded = asyncio.Event()
    release = asyncio.Event()

    async def waiting_session() -> AsyncIterator[MarketDataRecord]:
        yield tick(0)
        yielded.set()
        await release.wait()

    cfg = config(max_interval=timedelta(seconds=10))
    target, _, _, _ = persister(tmp_path, cfg)
    service = LiveIngestionService(
        adapter=FakeLiveAdapter([waiting_session]),
        persister=target,
        config=cfg,
    )
    task = asyncio.create_task(service.run(subscription(Timeframe.TICK)))
    await yielded.wait()
    await asyncio.sleep(0.01)
    service.request_stop()
    result = await asyncio.wait_for(task, timeout=1)
    release.set()

    assert result.written_batches == 1
    assert result.received_records == 1


@pytest.mark.asyncio
async def test_gap_recovery_uses_next_durable_tick_position(tmp_path: Path) -> None:
    gap = GapRecoveryPolicy(
        enabled=True,
        safety_lag=timedelta(seconds=1),
        min_gap=timedelta(0),
        chunk_size=timedelta(hours=1),
    )
    cfg = config(gap_recovery=gap)
    target, _, _, checkpoints = persister(tmp_path, cfg)
    checkpoints.advance(
        CheckpointKey("live-test", "fake", "EURUSD", DataKind.TICK),
        last_event_time=BASE,
        last_sequence=5,
        last_batch_id="prior",
    )
    recovery = FakeRecovery()
    service = LiveIngestionService(
        adapter=FakeLiveAdapter([session()]),
        persister=target,
        config=cfg,
        recovery=recovery,
        now=lambda: BASE + timedelta(seconds=10),
    )

    result = await service.run(subscription(Timeframe.TICK))
    assert result.recovery_runs == 1
    assert result.recovery_chunks == 0
    assert len(recovery.plans) == 1
    request = recovery.plans[0].request
    assert request.start == BASE + timedelta(microseconds=1)
    assert request.end == BASE + timedelta(seconds=9)


@pytest.mark.asyncio
async def test_gap_recovery_supports_initial_lookback_for_new_stream(tmp_path: Path) -> None:
    gap = GapRecoveryPolicy(
        enabled=True,
        safety_lag=timedelta(0),
        min_gap=timedelta(0),
        initial_lookback=timedelta(minutes=5),
    )
    cfg = config(gap_recovery=gap)
    target, _, _, _ = persister(tmp_path, cfg)
    recovery = FakeRecovery()
    service = LiveIngestionService(
        adapter=FakeLiveAdapter([session()]),
        persister=target,
        config=cfg,
        recovery=recovery,
        now=lambda: BASE,
    )

    await service.run(subscription(Timeframe.M1))
    assert recovery.plans[0].request.start == BASE - timedelta(minutes=5)
    assert recovery.plans[0].request.end == BASE


def test_gap_recovery_requires_backend_and_matching_config(tmp_path: Path) -> None:
    enabled = config(gap_recovery=GapRecoveryPolicy(enabled=True))
    target, _, _, _ = persister(tmp_path, enabled)
    adapter = FakeLiveAdapter([session()])
    with pytest.raises(LiveIngestionConfigurationError, match="no historical"):
        LiveIngestionService(adapter=adapter, persister=target, config=enabled)

    other = config(max_records=11)
    with pytest.raises(LiveIngestionConfigurationError, match="share one config"):
        LiveIngestionService(
            adapter=adapter,
            persister=target,
            config=other,
            recovery=FakeRecovery(),
        )


@pytest.mark.asyncio
async def test_non_transient_gap_recovery_failure_stops_service(tmp_path: Path) -> None:
    cfg = config(gap_recovery=GapRecoveryPolicy(enabled=True, initial_lookback=timedelta(minutes=1)))
    target, _, _, _ = persister(tmp_path, cfg)
    service = LiveIngestionService(
        adapter=FakeLiveAdapter([session()]),
        persister=target,
        config=cfg,
        recovery=FakeRecovery(error=RuntimeError("catalog broken")),
        now=lambda: BASE,
    )

    with pytest.raises(LiveGapRecoveryError, match="catalog broken"):
        await service.run(subscription(Timeframe.TICK))
