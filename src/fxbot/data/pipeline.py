"""Idempotent, chunked historical market-data ingestion coordination.

The coordinator composes an adapter, cleaner, quality gate, Parquet store,
metadata catalog, checkpoint store, and optional quarantine backend while
preserving strict commit ordering:

``fetch -> clean -> quality -> Parquet -> catalog -> checkpoint``

Each half-open chunk receives a deterministic content hash.  Repeating a plan
therefore reuses cataloged data rather than creating duplicate Parquet parts.
If a process crashes after catalog registration but before checkpoint advance,
the next run discovers the durable batch and completes only the checkpoint step.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from fxbot.data.adapters.base import AdapterDiagnostics, HistoricalMarketDataAdapter
from fxbot.data.catalog import (
    CatalogConflictError,
    DataCatalog,
    DatasetBatch,
    DatasetKey,
    DatasetRegistration,
)
from fxbot.data.checkpoints import CheckpointKey, CheckpointStore, IngestionCheckpoint
from fxbot.data.cleaning import CleanedBatch, CleaningReport, MarketDataCleaner
from fxbot.data.quality import DataQualityGate, QualityDecision
from fxbot.data.quarantine import JsonQuarantineStore, QuarantineEntry
from fxbot.data.storage import ParquetPartitionStore, StorageWriteResult
from fxbot.domain.enums import DataKind
from fxbot.domain.models import OHLC, Bar, HistoricalDataRequest, MarketDataRecord, Tick


class HistoricalIngestionError(RuntimeError):
    """Base class for historical ingestion coordination failures."""


class IngestionConflictError(HistoricalIngestionError):
    """Raised when a known content hash has incompatible catalog metadata."""


class IngestionPersistenceError(HistoricalIngestionError):
    """Raised when a persistence backend violates the coordinator contract."""


class IngestionCheckpointError(HistoricalIngestionError):
    """Raised after data is durable but checkpoint advancement fails."""

    def __init__(self, batch: DatasetBatch, cause: Exception) -> None:
        self.batch = batch
        self.cause = cause
        super().__init__(
            f"Batch {batch.batch_id!r} is durable, but checkpoint advancement failed: {cause}"
        )


class IngestionQualityError(HistoricalIngestionError):
    """Raised after a rejected chunk is optionally written to quarantine."""

    def __init__(
        self,
        request: HistoricalDataRequest,
        decision: QualityDecision,
        quarantine_file: Path | None,
    ) -> None:
        self.request = request
        self.decision = decision
        self.quarantine_file = quarantine_file
        detail = "; ".join(decision.reasons) or "unspecified quality failure"
        suffix = f"; quarantine={quarantine_file}" if quarantine_file is not None else ""
        super().__init__(f"Historical chunk rejected: {detail}{suffix}")


class EmptyChunkPolicy(StrEnum):
    """How the coordinator handles a chunk for which the adapter emits no data."""

    SKIP = "skip"
    REJECT = "reject"


class QualityRejectionPolicy(StrEnum):
    """Whether a quarantined quality failure stops or continues a backfill."""

    RAISE = "raise"
    QUARANTINE_CONTINUE = "quarantine_continue"


class IngestionDisposition(StrEnum):
    """Durable outcome of one half-open historical chunk."""

    WRITTEN = "written"
    ALREADY_REGISTERED = "already_registered"
    EMPTY_SKIPPED = "empty_skipped"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class HistoricalIngestionPlan:
    """Finite historical request split into deterministic half-open chunks."""

    pipeline_id: str
    source: str
    request: HistoricalDataRequest
    chunk_size: timedelta | None = None
    schema_version: int = 1
    empty_chunk_policy: EmptyChunkPolicy = EmptyChunkPolicy.SKIP
    quality_rejection_policy: QualityRejectionPolicy = QualityRejectionPolicy.RAISE
    quarantine_sample_size: int = 10

    def __post_init__(self) -> None:
        pipeline_id = self.pipeline_id.strip()
        source = self.source.strip()
        if not pipeline_id:
            raise ValueError("pipeline_id cannot be empty")
        if not source:
            raise ValueError("source cannot be empty")
        if self.request.start is None or self.request.end is None:
            raise ValueError("Historical ingestion requires bounded start and end times")
        if self.chunk_size is not None and self.chunk_size <= timedelta(0):
            raise ValueError("chunk_size must be positive")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.quarantine_sample_size < 0:
            raise ValueError("quarantine_sample_size must be non-negative")
        object.__setattr__(self, "pipeline_id", pipeline_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "empty_chunk_policy", EmptyChunkPolicy(self.empty_chunk_policy))
        object.__setattr__(
            self,
            "quality_rejection_policy",
            QualityRejectionPolicy(self.quality_rejection_policy),
        )


@dataclass(frozen=True, slots=True)
class HistoricalChunkResult:
    """Auditable outcome for one generated half-open request chunk."""

    request: HistoricalDataRequest
    disposition: IngestionDisposition
    diagnostics: AdapterDiagnostics
    cleaning_report: CleaningReport
    fetched_records: int
    cleaned_records: int
    quality: QualityDecision | None = None
    batch: DatasetBatch | None = None
    checkpoint: IngestionCheckpoint | None = None
    files_created: tuple[Path, ...] = ()
    content_hash: str | None = None
    quarantine_file: Path | None = None

    @property
    def wrote_new_data(self) -> bool:
        return self.disposition is IngestionDisposition.WRITTEN


@dataclass(frozen=True, slots=True)
class HistoricalIngestionResult:
    """Aggregate result of executing every chunk in one ingestion plan."""

    plan: HistoricalIngestionPlan
    chunks: tuple[HistoricalChunkResult, ...]

    @property
    def fetched_records(self) -> int:
        return sum(item.fetched_records for item in self.chunks)

    @property
    def cleaned_records(self) -> int:
        return sum(item.cleaned_records for item in self.chunks)

    @property
    def written_batches(self) -> int:
        return sum(item.disposition is IngestionDisposition.WRITTEN for item in self.chunks)

    @property
    def reused_batches(self) -> int:
        return sum(
            item.disposition is IngestionDisposition.ALREADY_REGISTERED
            for item in self.chunks
        )

    @property
    def skipped_chunks(self) -> int:
        return sum(
            item.disposition is IngestionDisposition.EMPTY_SKIPPED for item in self.chunks
        )

    @property
    def quarantined_chunks(self) -> int:
        return sum(item.disposition is IngestionDisposition.QUARANTINED for item in self.chunks)

    @property
    def files_created(self) -> tuple[Path, ...]:
        return tuple(path for chunk in self.chunks for path in chunk.files_created)


class MarketDataStore(Protocol):
    """Structural contract required from the durable record store."""

    def append(self, records: Iterable[MarketDataRecord]) -> StorageWriteResult: ...


class CatalogStore(Protocol):
    """Structural contract required from the metadata catalog."""

    def get_by_hash(self, content_hash: str) -> DatasetBatch | None: ...

    def register_batch(
        self,
        registration: DatasetRegistration,
        *,
        batch_id: str | None = None,
    ) -> DatasetBatch: ...


class CheckpointBackend(Protocol):
    """Structural contract required from the checkpoint store."""

    def get(self, key: CheckpointKey) -> IngestionCheckpoint | None: ...

    def advance(
        self,
        key: CheckpointKey,
        *,
        last_event_time: datetime,
        last_sequence: int | None = None,
        last_batch_id: str | None = None,
        updated_at: datetime | None = None,
        allow_regression: bool = False,
    ) -> IngestionCheckpoint: ...


class QuarantineBackend(Protocol):
    """Structural contract required from a rejected-batch audit store."""

    def write(self, entry: QuarantineEntry) -> Path: ...


class HistoricalIngestionCoordinator:
    """Run bounded historical backfills with quality and idempotency controls."""

    def __init__(
        self,
        *,
        adapter: HistoricalMarketDataAdapter,
        cleaner: MarketDataCleaner,
        quality_gate: DataQualityGate,
        storage: MarketDataStore,
        catalog: CatalogStore,
        checkpoints: CheckpointBackend,
        quarantine: QuarantineBackend | None = None,
    ) -> None:
        self.adapter = adapter
        self.cleaner = cleaner
        self.quality_gate = quality_gate
        self.storage = storage
        self.catalog = catalog
        self.checkpoints = checkpoints
        self.quarantine = quarantine

    @classmethod
    def from_defaults(
        cls,
        *,
        adapter: HistoricalMarketDataAdapter,
        storage: ParquetPartitionStore,
        catalog: DataCatalog,
        checkpoints: CheckpointStore,
        quarantine_root: str | Path | None = None,
    ) -> HistoricalIngestionCoordinator:
        """Construct a coordinator with default cleaner and quality policy."""

        return cls(
            adapter=adapter,
            cleaner=MarketDataCleaner(),
            quality_gate=DataQualityGate(),
            storage=storage,
            catalog=catalog,
            checkpoints=checkpoints,
            quarantine=(
                JsonQuarantineStore(quarantine_root)
                if quarantine_root is not None
                else None
            ),
        )

    def run(self, plan: HistoricalIngestionPlan) -> HistoricalIngestionResult:
        """Execute every generated chunk in chronological order."""

        results: list[HistoricalChunkResult] = []
        for request in iter_chunk_requests(plan):
            result = self._run_chunk(plan, request)
            results.append(result)
        return HistoricalIngestionResult(plan=plan, chunks=tuple(results))

    def _run_chunk(
        self,
        plan: HistoricalIngestionPlan,
        request: HistoricalDataRequest,
    ) -> HistoricalChunkResult:
        fetched = self._fetch(request)
        diagnostics = self.adapter.diagnostics
        self._validate_records(fetched, request)
        cleaned = self._clean(fetched, request.kind)

        if not fetched and plan.empty_chunk_policy is EmptyChunkPolicy.SKIP:
            return HistoricalChunkResult(
                request=request,
                disposition=IngestionDisposition.EMPTY_SKIPPED,
                diagnostics=diagnostics,
                cleaning_report=cleaned.report,
                fetched_records=0,
                cleaned_records=0,
            )

        quality = self.quality_gate.evaluate(cleaned.report, diagnostics)
        if not quality.accepted:
            quarantine_file = self._quarantine(
                plan,
                request,
                fetched=fetched,
                diagnostics=diagnostics,
                report=cleaned.report,
                decision=quality,
            )
            if plan.quality_rejection_policy is QualityRejectionPolicy.RAISE:
                raise IngestionQualityError(request, quality, quarantine_file)
            return HistoricalChunkResult(
                request=request,
                disposition=IngestionDisposition.QUARANTINED,
                diagnostics=diagnostics,
                cleaning_report=cleaned.report,
                fetched_records=len(fetched),
                cleaned_records=len(cleaned.records),
                quality=quality,
                quarantine_file=quarantine_file,
            )

        records = cleaned.records
        if not records:
            raise HistoricalIngestionError("Cannot persist an empty historical batch")

        content_hash = compute_content_hash(
            records,
            source=plan.source,
            schema_version=plan.schema_version,
        )
        key = DatasetKey(
            kind=request.kind,
            symbol=request.symbol,
            timeframe=request.timeframe,
        )
        start_time, end_time = _coverage(records)
        checkpoint_time, checkpoint_sequence = _checkpoint_position(records)

        existing = self.catalog.get_by_hash(content_hash)
        if existing is not None:
            self._verify_existing(
                existing,
                key=key,
                start_time=start_time,
                end_time=end_time,
                record_count=len(records),
                source=plan.source,
                schema_version=plan.schema_version,
            )
            checkpoint = self._advance_checkpoint(
                plan,
                request=request,
                batch=existing,
                event_time=checkpoint_time,
                sequence=checkpoint_sequence,
            )
            return HistoricalChunkResult(
                request=request,
                disposition=IngestionDisposition.ALREADY_REGISTERED,
                batch=existing,
                checkpoint=checkpoint,
                quality=quality,
                diagnostics=diagnostics,
                cleaning_report=cleaned.report,
                fetched_records=len(fetched),
                cleaned_records=len(records),
                content_hash=content_hash,
            )

        write_result = self.storage.append(records)
        self._validate_write_result(write_result, len(records))
        registration = DatasetRegistration(
            key=key,
            start_time=start_time,
            end_time=end_time,
            record_count=len(records),
            source=plan.source,
            schema_version=plan.schema_version,
            content_hash=content_hash,
            files=write_result.files_created,
        )

        try:
            batch = self.catalog.register_batch(registration)
        except CatalogConflictError:
            raced = self.catalog.get_by_hash(content_hash)
            if raced is None:
                self._remove_files(write_result.files_created)
                raise
            try:
                self._verify_existing(
                    raced,
                    key=key,
                    start_time=start_time,
                    end_time=end_time,
                    record_count=len(records),
                    source=plan.source,
                    schema_version=plan.schema_version,
                )
            except Exception:
                self._remove_files(write_result.files_created)
                raise
            self._remove_files(write_result.files_created)
            batch = raced
            disposition = IngestionDisposition.ALREADY_REGISTERED
            created_files: tuple[Path, ...] = ()
        except Exception:
            self._remove_files(write_result.files_created)
            raise
        else:
            disposition = IngestionDisposition.WRITTEN
            created_files = write_result.files_created

        checkpoint = self._advance_checkpoint(
            plan,
            request=request,
            batch=batch,
            event_time=checkpoint_time,
            sequence=checkpoint_sequence,
        )
        return HistoricalChunkResult(
            request=request,
            disposition=disposition,
            batch=batch,
            checkpoint=checkpoint,
            quality=quality,
            diagnostics=diagnostics,
            cleaning_report=cleaned.report,
            fetched_records=len(fetched),
            cleaned_records=len(records),
            files_created=created_files,
            content_hash=content_hash,
        )

    def _fetch(self, request: HistoricalDataRequest) -> tuple[MarketDataRecord, ...]:
        if request.kind is DataKind.TICK:
            return tuple(self.adapter.iter_ticks(request))
        return tuple(self.adapter.iter_bars(request))

    def _clean(
        self,
        records: tuple[MarketDataRecord, ...],
        kind: DataKind,
    ) -> CleanedBatch[Tick] | CleanedBatch[Bar]:
        if kind is DataKind.TICK:
            ticks = tuple(record for record in records if isinstance(record, Tick))
            return self.cleaner.clean_ticks(ticks)
        bars = tuple(record for record in records if isinstance(record, Bar))
        return self.cleaner.clean_bars(bars)

    @staticmethod
    def _validate_records(
        records: tuple[MarketDataRecord, ...],
        request: HistoricalDataRequest,
    ) -> None:
        for record in records:
            if record.symbol != request.symbol:
                raise HistoricalIngestionError(
                    f"Adapter emitted {record.symbol}, expected {request.symbol}"
                )
            if request.kind is DataKind.TICK:
                if not isinstance(record, Tick):
                    raise HistoricalIngestionError("Tick request emitted a bar record")
                timestamp = record.event_time
            else:
                if not isinstance(record, Bar):
                    raise HistoricalIngestionError("Bar request emitted a tick record")
                if record.timeframe is not request.timeframe:
                    raise HistoricalIngestionError(
                        f"Adapter emitted timeframe {record.timeframe}, "
                        f"expected {request.timeframe}"
                    )
                timestamp = record.open_time
            if not request.contains(timestamp):
                raise HistoricalIngestionError(
                    f"Adapter emitted record outside requested range: {timestamp.isoformat()}"
                )

    @staticmethod
    def _validate_write_result(result: StorageWriteResult, expected_count: int) -> None:
        if result.records_written != expected_count:
            HistoricalIngestionCoordinator._remove_files(result.files_created)
            raise IngestionPersistenceError(
                f"Storage reported {result.records_written} records, expected {expected_count}"
            )
        if not result.files_created:
            raise IngestionPersistenceError("Storage did not create any durable files")

    @staticmethod
    def _verify_existing(
        batch: DatasetBatch,
        *,
        key: DatasetKey,
        start_time: datetime,
        end_time: datetime,
        record_count: int,
        source: str,
        schema_version: int,
    ) -> None:
        compatible = (
            batch.key == key
            and batch.start_time == start_time
            and batch.end_time == end_time
            and batch.record_count == record_count
            and batch.source == source
            and batch.schema_version == schema_version
        )
        if not compatible:
            raise IngestionConflictError(
                f"Content hash {batch.content_hash!r} is registered with incompatible metadata"
            )

    def _advance_checkpoint(
        self,
        plan: HistoricalIngestionPlan,
        *,
        request: HistoricalDataRequest,
        batch: DatasetBatch,
        event_time: datetime,
        sequence: int | None,
    ) -> IngestionCheckpoint:
        checkpoint_key = CheckpointKey(
            pipeline_id=plan.pipeline_id,
            source=plan.source,
            symbol=request.symbol,
            kind=request.kind,
            timeframe=request.timeframe,
        )
        current = self.checkpoints.get(checkpoint_key)
        if current is not None and _checkpoint_sort_key(
            current.last_event_time,
            current.last_sequence,
        ) >= _checkpoint_sort_key(event_time, sequence):
            return current
        try:
            return self.checkpoints.advance(
                checkpoint_key,
                last_event_time=event_time,
                last_sequence=sequence,
                last_batch_id=batch.batch_id,
            )
        except Exception as exc:
            raise IngestionCheckpointError(batch, exc) from exc

    def _quarantine(
        self,
        plan: HistoricalIngestionPlan,
        request: HistoricalDataRequest,
        *,
        fetched: tuple[MarketDataRecord, ...],
        diagnostics: AdapterDiagnostics,
        report: CleaningReport,
        decision: QualityDecision,
    ) -> Path | None:
        if self.quarantine is None:
            return None
        sample = tuple(
            _record_payload(record)
            for record in fetched[: plan.quarantine_sample_size]
        )
        assert request.start is not None
        assert request.end is not None
        return self.quarantine.write(
            QuarantineEntry(
                pipeline_id=plan.pipeline_id,
                source=plan.source,
                symbol=request.symbol,
                kind=request.kind,
                timeframe=request.timeframe,
                start_time=request.start,
                end_time=request.end,
                fetched_records=len(fetched),
                diagnostics=diagnostics,
                cleaning_report=report,
                decision=decision,
                sample_records=sample,
            )
        )

    @staticmethod
    def _remove_files(files: Iterable[Path]) -> None:
        for path in files:
            # Preserve the primary catalog/storage exception. Any remaining
            # orphan is discoverable and can be reconciled operationally.
            with suppress(OSError):
                Path(path).unlink(missing_ok=True)


def iter_chunk_requests(plan: HistoricalIngestionPlan) -> Iterator[HistoricalDataRequest]:
    """Yield non-overlapping half-open requests covering the complete plan."""

    start = plan.request.start
    end = plan.request.end
    assert start is not None
    assert end is not None
    if plan.chunk_size is None:
        yield plan.request
        return

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + plan.chunk_size, end)
        yield HistoricalDataRequest(
            symbol=plan.request.symbol,
            kind=plan.request.kind,
            start=cursor,
            end=chunk_end,
            timeframe=plan.request.timeframe,
        )
        cursor = chunk_end


def compute_content_hash(
    records: Iterable[MarketDataRecord],
    *,
    source: str,
    schema_version: int,
) -> str:
    """Return a deterministic SHA-256 digest for canonical market-data records."""

    normalized_source = source.strip()
    if not normalized_source:
        raise ValueError("source cannot be empty")
    if schema_version <= 0:
        raise ValueError("schema_version must be positive")

    digest = hashlib.sha256()
    header = {
        "hash_schema": 1,
        "source": normalized_source,
        "storage_schema": schema_version,
    }
    digest.update(_json_bytes(header))
    count = 0
    for record in records:
        digest.update(b"\n")
        digest.update(_json_bytes(_record_payload(record)))
        count += 1
    if count == 0:
        raise ValueError("records cannot be empty")
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _record_payload(record: MarketDataRecord) -> dict[str, object]:
    if isinstance(record, Tick):
        return {
            "kind": "tick",
            "symbol": record.symbol,
            "event_time": _time_text(record.event_time),
            "bid": record.bid.hex(),
            "ask": record.ask.hex(),
            "bid_size": _optional_float(record.bid_size),
            "ask_size": _optional_float(record.ask_size),
            "source": record.source,
            "sequence": record.sequence,
            "received_time": (
                _time_text(record.received_time) if record.received_time is not None else None
            ),
        }
    return {
        "kind": "bar",
        "symbol": record.symbol,
        "open_time": _time_text(record.open_time),
        "timeframe": record.timeframe.value,
        "bid": _ohlc_payload(record.bid),
        "ask": _ohlc_payload(record.ask),
        "mid": _ohlc_payload(record.mid_ohlc) if record.mid_ohlc is not None else None,
        "tick_volume": record.tick_volume,
        "real_volume": _optional_float(record.real_volume),
        "source": record.source,
        "complete": record.complete,
    }


def _ohlc_payload(value: OHLC) -> tuple[str, str, str, str]:
    return (value.open.hex(), value.high.hex(), value.low.hex(), value.close.hex())


def _optional_float(value: float | None) -> str | None:
    return float(value).hex() if value is not None else None


def _time_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _coverage(records: tuple[Tick, ...] | tuple[Bar, ...]) -> tuple[datetime, datetime]:
    if isinstance(records[0], Tick):
        ticks = tuple(record for record in records if isinstance(record, Tick))
        start = min(record.event_time for record in ticks)
        end = max(record.event_time for record in ticks) + timedelta(microseconds=1)
        return start, end
    bars = tuple(record for record in records if isinstance(record, Bar))
    return min(record.open_time for record in bars), max(record.close_time for record in bars)


def _checkpoint_position(
    records: tuple[Tick, ...] | tuple[Bar, ...],
) -> tuple[datetime, int | None]:
    if isinstance(records[0], Tick):
        ticks = tuple(record for record in records if isinstance(record, Tick))
        event_time = max(record.event_time for record in ticks)
        sequences = [
            record.sequence
            for record in ticks
            if record.event_time == event_time and record.sequence is not None
        ]
        return event_time, max(sequences) if sequences else None
    bars = tuple(record for record in records if isinstance(record, Bar))
    return max(record.open_time for record in bars), None


def _checkpoint_sort_key(event_time: datetime, sequence: int | None) -> tuple[datetime, int]:
    return event_time, sequence if sequence is not None else -1
