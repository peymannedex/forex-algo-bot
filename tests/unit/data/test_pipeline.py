from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fxbot.data.adapters.base import AdapterDiagnostics, HistoricalMarketDataAdapter
from fxbot.data.catalog import DataCatalog, DatasetKey, DatasetRegistration
from fxbot.data.checkpoints import CheckpointKey, CheckpointStore
from fxbot.data.cleaning import MarketDataCleaner
from fxbot.data.pipeline import (
    EmptyChunkPolicy,
    HistoricalIngestionCoordinator,
    HistoricalIngestionError,
    HistoricalIngestionPlan,
    IngestionCheckpointError,
    IngestionConflictError,
    IngestionDisposition,
    IngestionPersistenceError,
    IngestionQualityError,
    QualityRejectionPolicy,
    compute_content_hash,
    iter_chunk_requests,
)
from fxbot.data.quality import DataQualityGate, QualityThresholds
from fxbot.data.quarantine import JsonQuarantineStore
from fxbot.data.storage import PartitionRef, StorageWriteResult
from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import OHLC, Bar, HistoricalDataRequest, MarketDataRecord, Tick

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


def bar(minute: int, *, timeframe: Timeframe = Timeframe.M1) -> Bar:
    value = 1.10 + minute * 0.001
    return Bar(
        symbol="EURUSD",
        open_time=BASE + timedelta(minutes=minute),
        timeframe=timeframe,
        bid=OHLC(value, value + 0.001, value - 0.001, value + 0.0005),
        ask=OHLC(value + 0.0002, value + 0.0012, value - 0.0008, value + 0.0007),
        tick_volume=10,
        source="fake",
    )


class FakeAdapter(HistoricalMarketDataAdapter):
    def __init__(
        self,
        records: tuple[MarketDataRecord, ...],
        *,
        honor_request: bool = True,
    ) -> None:
        self.records = records
        self.honor_request = honor_request
        self._diagnostics = AdapterDiagnostics()

    @property
    def diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics

    def iter_ticks(self, request: HistoricalDataRequest) -> Iterator[Tick]:
        ticks = tuple(record for record in self.records if isinstance(record, Tick))
        emitted = (
            tuple(record for record in ticks if request.contains(record.event_time))
            if self.honor_request
            else ticks
        )
        self._diagnostics = AdapterDiagnostics(
            rows_read=len(emitted),
            records_emitted=len(emitted),
        )
        yield from emitted

    def iter_bars(self, request: HistoricalDataRequest) -> Iterator[Bar]:
        bars = tuple(record for record in self.records if isinstance(record, Bar))
        emitted = (
            tuple(record for record in bars if request.contains(record.open_time))
            if self.honor_request
            else bars
        )
        self._diagnostics = AdapterDiagnostics(
            rows_read=len(emitted),
            records_emitted=len(emitted),
        )
        yield from emitted


class FileStorage:
    def __init__(self, root: Path, *, reported_count: int | None = None) -> None:
        self.root = root
        self.reported_count = reported_count
        self.calls = 0

    def append(self, records) -> StorageWriteResult:  # type: ignore[no-untyped-def]
        items = tuple(records)
        self.calls += 1
        path = self.root / f"part-{self.calls}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-parquet")
        first = items[0]
        partition = (
            PartitionRef(DataKind.TICK, first.symbol, 2026, 1)
            if isinstance(first, Tick)
            else PartitionRef(DataKind.BAR, first.symbol, 2026, 1, first.timeframe)
        )
        return StorageWriteResult(
            records_written=(
                self.reported_count if self.reported_count is not None else len(items)
            ),
            files_created=(path,),
            partitions=(partition,),
        )


class FailingCheckpointStore:
    def get(self, key: CheckpointKey):  # type: ignore[no-untyped-def]
        return None

    def advance(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("checkpoint unavailable")


class FailingCatalog:
    def get_by_hash(self, content_hash: str):  # type: ignore[no-untyped-def]
        return None

    def register_batch(self, registration, *, batch_id=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("catalog unavailable")


def request(
    kind: DataKind = DataKind.TICK,
    *,
    start: datetime = BASE,
    end: datetime = BASE + timedelta(hours=1),
) -> HistoricalDataRequest:
    return HistoricalDataRequest(
        symbol="EURUSD",
        kind=kind,
        timeframe=Timeframe.M1 if kind is DataKind.BAR else None,
        start=start,
        end=end,
    )


def plan(
    kind: DataKind = DataKind.TICK,
    *,
    start: datetime = BASE,
    end: datetime = BASE + timedelta(hours=1),
    chunk_size: timedelta | None = None,
    empty_chunk_policy: EmptyChunkPolicy = EmptyChunkPolicy.SKIP,
    quality_rejection_policy: QualityRejectionPolicy = QualityRejectionPolicy.RAISE,
) -> HistoricalIngestionPlan:
    return HistoricalIngestionPlan(
        "historical-test",
        "fake",
        request(kind, start=start, end=end),
        chunk_size=chunk_size,
        empty_chunk_policy=empty_chunk_policy,
        quality_rejection_policy=quality_rejection_policy,
    )


def coordinator(
    tmp_path: Path,
    records: tuple[MarketDataRecord, ...],
    *,
    storage: FileStorage | None = None,
    catalog=None,  # type: ignore[no-untyped-def]
    checkpoints=None,  # type: ignore[no-untyped-def]
    gate: DataQualityGate | None = None,
    honor_request: bool = True,
    quarantine: bool = False,
) -> tuple[HistoricalIngestionCoordinator, FileStorage, DataCatalog, CheckpointStore]:
    actual_storage = storage or FileStorage(tmp_path / "parts")
    actual_catalog = catalog or DataCatalog(tmp_path / "state.sqlite3")
    actual_checkpoints = checkpoints or CheckpointStore(tmp_path / "state.sqlite3")
    return (
        HistoricalIngestionCoordinator(
            adapter=FakeAdapter(records, honor_request=honor_request),
            cleaner=MarketDataCleaner(),
            quality_gate=gate or DataQualityGate(),
            storage=actual_storage,
            catalog=actual_catalog,
            checkpoints=actual_checkpoints,
            quarantine=(
                JsonQuarantineStore(tmp_path / "quarantine") if quarantine else None
            ),
        ),
        actual_storage,
        actual_catalog,
        actual_checkpoints,
    )


def test_historical_plan_and_chunking_require_valid_metadata() -> None:
    with pytest.raises(ValueError, match="pipeline_id"):
        HistoricalIngestionPlan("", "fake", request())
    with pytest.raises(ValueError, match="source"):
        HistoricalIngestionPlan("pipeline", " ", request())
    with pytest.raises(ValueError, match="bounded"):
        HistoricalIngestionPlan(
            "pipeline",
            "fake",
            HistoricalDataRequest("EURUSD", DataKind.TICK),
        )
    with pytest.raises(ValueError, match="chunk_size"):
        plan(chunk_size=timedelta(0))

    chunks = tuple(
        iter_chunk_requests(
            plan(
                end=BASE + timedelta(hours=2, minutes=30),
                chunk_size=timedelta(hours=1),
            )
        )
    )
    assert [(item.start, item.end) for item in chunks] == [
        (BASE, BASE + timedelta(hours=1)),
        (BASE + timedelta(hours=1), BASE + timedelta(hours=2)),
        (BASE + timedelta(hours=2), BASE + timedelta(hours=2, minutes=30)),
    ]


def test_tick_ingestion_writes_catalog_and_checkpoint(tmp_path) -> None:
    records = (tick(0, sequence=1), tick(1, sequence=2))
    service, storage, catalog, checkpoints = coordinator(tmp_path, records)

    result = service.run(plan())
    chunk = result.chunks[0]

    assert chunk.disposition is IngestionDisposition.WRITTEN
    assert chunk.wrote_new_data
    assert result.fetched_records == 2
    assert result.cleaned_records == 2
    assert result.written_batches == 1
    assert storage.calls == 1
    assert chunk.batch is not None
    assert catalog.get_batch(chunk.batch.batch_id) == chunk.batch
    assert chunk.batch.start_time == BASE
    assert chunk.batch.end_time == BASE + timedelta(seconds=1, microseconds=1)
    key = CheckpointKey("historical-test", "fake", "EURUSD", DataKind.TICK)
    saved = checkpoints.get(key)
    assert saved == chunk.checkpoint
    assert saved is not None and saved.last_sequence == 2


def test_chunk_boundaries_are_half_open_and_non_overlapping(tmp_path) -> None:
    records = (
        tick(0, sequence=1),
        tick(3600, sequence=2),
        tick(7200, sequence=3),
    )
    service, storage, catalog, _ = coordinator(tmp_path, records)

    result = service.run(
        plan(
            end=BASE + timedelta(hours=3),
            chunk_size=timedelta(hours=1),
        )
    )

    assert [chunk.fetched_records for chunk in result.chunks] == [1, 1, 1]
    assert result.written_batches == 3
    assert storage.calls == 3
    assert len(catalog.list_batches()) == 3


def test_repeated_chunked_plan_is_idempotent_without_checkpoint_regression(tmp_path) -> None:
    records = (tick(0, sequence=1), tick(3600, sequence=2))
    service, storage, catalog, checkpoints = coordinator(tmp_path, records)
    ingestion_plan = plan(
        end=BASE + timedelta(hours=2),
        chunk_size=timedelta(hours=1),
    )

    first = service.run(ingestion_plan)
    second = service.run(ingestion_plan)

    assert first.written_batches == 2
    assert second.reused_batches == 2
    assert storage.calls == 2
    assert len(catalog.list_batches()) == 2
    assert len(checkpoints.list()) == 1
    assert second.chunks[0].checkpoint == second.chunks[1].checkpoint


def test_empty_chunks_are_skipped_without_persistence(tmp_path) -> None:
    service, storage, catalog, checkpoints = coordinator(tmp_path, ())

    result = service.run(plan())

    assert result.skipped_chunks == 1
    assert result.chunks[0].disposition is IngestionDisposition.EMPTY_SKIPPED
    assert storage.calls == 0
    assert catalog.list_batches() == ()
    assert checkpoints.list() == ()


def test_quality_rejection_is_quarantined_before_raise(tmp_path) -> None:
    gate = DataQualityGate(QualityThresholds(min_output_records=2))
    service, storage, catalog, checkpoints = coordinator(
        tmp_path,
        (tick(0),),
        gate=gate,
        quarantine=True,
    )

    with pytest.raises(IngestionQualityError) as captured:
        service.run(plan())

    assert captured.value.quarantine_file is not None
    assert captured.value.quarantine_file.exists()
    assert storage.calls == 0
    assert catalog.list_batches() == ()
    assert checkpoints.list() == ()


def test_quarantine_continue_processes_later_healthy_chunks(tmp_path) -> None:
    records = (
        tick(0),
        tick(3600),
        tick(3601),
    )
    gate = DataQualityGate(QualityThresholds(min_output_records=2))
    service, storage, catalog, _ = coordinator(
        tmp_path,
        records,
        gate=gate,
        quarantine=True,
    )

    result = service.run(
        plan(
            end=BASE + timedelta(hours=2),
            chunk_size=timedelta(hours=1),
            quality_rejection_policy=QualityRejectionPolicy.QUARANTINE_CONTINUE,
        )
    )

    assert [item.disposition for item in result.chunks] == [
        IngestionDisposition.QUARANTINED,
        IngestionDisposition.WRITTEN,
    ]
    assert result.quarantined_chunks == 1
    assert result.written_batches == 1
    assert storage.calls == 1
    assert len(catalog.list_batches()) == 1


def test_empty_reject_policy_uses_quality_gate_and_quarantine(tmp_path) -> None:
    service, _, _, _ = coordinator(tmp_path, (), quarantine=True)

    with pytest.raises(IngestionQualityError) as captured:
        service.run(plan(empty_chunk_policy=EmptyChunkPolicy.REJECT))

    assert captured.value.quarantine_file is not None
    assert captured.value.decision.metrics.output_records == 0


def test_adapter_contract_violations_are_rejected_before_storage(tmp_path) -> None:
    wrong_symbol, storage, catalog, _ = coordinator(
        tmp_path,
        (tick(0, symbol="GBPUSD"),),
        honor_request=False,
    )
    with pytest.raises(HistoricalIngestionError, match="expected EURUSD"):
        wrong_symbol.run(plan())
    assert storage.calls == 0
    assert catalog.list_batches() == ()

    outside, outside_storage, _, _ = coordinator(
        tmp_path / "outside",
        (tick(3600),),
        honor_request=False,
    )
    with pytest.raises(HistoricalIngestionError, match="outside requested range"):
        outside.run(plan())
    assert outside_storage.calls == 0


def test_bar_ingestion_uses_close_time_coverage_and_bar_checkpoint(tmp_path) -> None:
    records = (bar(0), bar(1))
    service, storage, catalog, checkpoints = coordinator(tmp_path, records)

    chunk = service.run(plan(DataKind.BAR)).chunks[0]

    assert chunk.batch is not None
    assert chunk.batch.key == DatasetKey(DataKind.BAR, "EURUSD", Timeframe.M1)
    assert chunk.batch.start_time == BASE
    assert chunk.batch.end_time == BASE + timedelta(minutes=2)
    assert chunk.checkpoint is not None
    assert chunk.checkpoint.last_event_time == BASE + timedelta(minutes=1)
    assert chunk.checkpoint.last_sequence is None
    assert storage.calls == 1
    assert len(catalog.list_batches()) == 1
    assert len(checkpoints.list()) == 1


def test_wrong_bar_timeframe_is_rejected(tmp_path) -> None:
    service, storage, _, _ = coordinator(
        tmp_path,
        (bar(0, timeframe=Timeframe.M5),),
        honor_request=False,
    )

    with pytest.raises(HistoricalIngestionError, match="expected 1m"):
        service.run(plan(DataKind.BAR))

    assert storage.calls == 0


def test_storage_count_mismatch_removes_created_file(tmp_path) -> None:
    storage = FileStorage(tmp_path / "parts", reported_count=99)
    service, _, catalog, checkpoints = coordinator(
        tmp_path,
        (tick(0), tick(1)),
        storage=storage,
    )

    with pytest.raises(IngestionPersistenceError, match="expected 2"):
        service.run(plan())

    assert list((tmp_path / "parts").glob("*.parquet")) == []
    assert catalog.list_batches() == ()
    assert checkpoints.list() == ()


def test_catalog_failure_removes_new_parquet_files(tmp_path) -> None:
    storage = FileStorage(tmp_path / "parts")
    service, _, _, checkpoints = coordinator(
        tmp_path,
        (tick(0),),
        storage=storage,
        catalog=FailingCatalog(),
    )

    with pytest.raises(RuntimeError, match="catalog unavailable"):
        service.run(plan())

    assert list((tmp_path / "parts").glob("*.parquet")) == []
    assert checkpoints.list() == ()


def test_checkpoint_failure_preserves_durable_batch_and_retry_recovers(tmp_path) -> None:
    records = (tick(0, sequence=1), tick(1, sequence=2))
    storage = FileStorage(tmp_path / "parts")
    catalog = DataCatalog(tmp_path / "state.sqlite3")
    failed, _, _, _ = coordinator(
        tmp_path,
        records,
        storage=storage,
        catalog=catalog,
        checkpoints=FailingCheckpointStore(),
    )

    with pytest.raises(IngestionCheckpointError) as captured:
        failed.run(plan())

    durable = captured.value.batch
    assert catalog.get_batch(durable.batch_id) == durable
    assert storage.calls == 1

    checkpoints = CheckpointStore(tmp_path / "state.sqlite3")
    retry, _, _, _ = coordinator(
        tmp_path,
        records,
        storage=storage,
        catalog=catalog,
        checkpoints=checkpoints,
    )
    recovered = retry.run(plan()).chunks[0]

    assert recovered.disposition is IngestionDisposition.ALREADY_REGISTERED
    assert storage.calls == 1
    assert recovered.checkpoint is not None
    assert recovered.checkpoint.last_batch_id == durable.batch_id


def test_existing_hash_with_incompatible_metadata_is_rejected(tmp_path) -> None:
    records = (tick(0), tick(1))
    digest = compute_content_hash(records, source="fake", schema_version=1)
    catalog = DataCatalog(tmp_path / "state.sqlite3")
    catalog.register_batch(
        DatasetRegistration(
            key=DatasetKey(DataKind.TICK, "GBPUSD"),
            start_time=BASE,
            end_time=BASE + timedelta(seconds=1, microseconds=1),
            record_count=2,
            source="fake",
            schema_version=1,
            content_hash=digest,
            files=(Path("winner.parquet"),),
        )
    )
    service, storage, _, checkpoints = coordinator(
        tmp_path,
        records,
        catalog=catalog,
    )

    with pytest.raises(IngestionConflictError, match="incompatible"):
        service.run(plan())

    assert storage.calls == 0
    assert checkpoints.list() == ()


def test_content_hash_is_deterministic_and_sensitive_to_provenance() -> None:
    records = (tick(0), tick(1))

    first = compute_content_hash(records, source="fake", schema_version=1)
    second = compute_content_hash(records, source="fake", schema_version=1)

    assert first == second
    assert first != compute_content_hash(tuple(reversed(records)), source="fake", schema_version=1)
    assert first != compute_content_hash(records, source="other", schema_version=1)
    assert first != compute_content_hash(records, source="fake", schema_version=2)
    changed = (replace(records[0], bid=1.09), records[1])
    assert first != compute_content_hash(changed, source="fake", schema_version=1)
    with pytest.raises(ValueError, match="records cannot be empty"):
        compute_content_hash((), source="fake", schema_version=1)
