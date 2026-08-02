"""Asynchronous live market-data ingestion with recovery and micro-batching.

The service keeps the event loop responsive while composing the existing live
adapter, cleaner, quality gate, Parquet store, catalog, and checkpoint store.
Each stream is buffered independently and committed in this order:

``clean -> quality -> Parquet -> catalog -> checkpoint``

Reconnects use bounded exponential backoff.  Before a restarted live session,
an optional historical coordinator fills the interval after the last durable
checkpoint, preventing ordinary connection gaps from becoming silent holes.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from fxbot.data.adapters.base import (
    AdapterDiagnostics,
    LiveMarketDataAdapter,
    MarketDataAdapterError,
)
from fxbot.data.catalog import (
    CatalogConflictError,
    DatasetBatch,
    DatasetKey,
    DatasetRegistration,
)
from fxbot.data.checkpoints import CheckpointKey, IngestionCheckpoint
from fxbot.data.cleaning import CleanedBatch, CleaningReport, MarketDataCleaner
from fxbot.data.pipeline import (
    HistoricalIngestionPlan,
    HistoricalIngestionResult,
    IngestionConflictError,
    QualityRejectionPolicy,
    compute_content_hash,
)
from fxbot.data.quality import DataQualityGate, QualityDecision
from fxbot.data.quarantine import QuarantineEntry
from fxbot.data.retry import ReconnectPolicy
from fxbot.data.storage import StorageWriteResult
from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import (
    Bar,
    HistoricalDataRequest,
    LiveSubscription,
    MarketDataRecord,
    Tick,
)


class LiveIngestionError(RuntimeError):
    """Base class for non-transient live-ingestion failures."""


class LiveIngestionConfigurationError(LiveIngestionError):
    """Raised when service dependencies do not match enabled features."""


class LiveStreamEndedError(MarketDataAdapterError):
    """Raised internally when a live stream ends without a stop request."""


class LiveReconnectExhaustedError(LiveIngestionError):
    """Raised after the reconnect policy denies another transient retry."""

    def __init__(self, attempts: int, cause: Exception) -> None:
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"Live reconnect attempts exhausted after {attempts} retries: {cause}")


class LiveIngestionQualityError(LiveIngestionError):
    """Raised when a micro-batch fails the configured quality gate."""

    def __init__(self, decision: QualityDecision, quarantine_file: Path | None) -> None:
        self.decision = decision
        self.quarantine_file = quarantine_file
        details = "; ".join(decision.reasons) or "unspecified quality failure"
        suffix = f"; quarantine={quarantine_file}" if quarantine_file is not None else ""
        super().__init__(f"Live micro-batch rejected: {details}{suffix}")


class LivePersistenceError(LiveIngestionError):
    """Raised when storage or catalog persistence violates its contract."""


class LiveCheckpointError(LiveIngestionError):
    """Raised after a live batch is durable but checkpoint advancement fails."""

    def __init__(self, batch: DatasetBatch, cause: Exception) -> None:
        self.batch = batch
        self.cause = cause
        super().__init__(
            f"Live batch {batch.batch_id!r} is durable, but checkpoint advancement failed: {cause}"
        )


class LiveGapRecoveryError(LiveIngestionError):
    """Raised when a non-transient historical gap-recovery run fails."""


class LiveBatchDisposition(StrEnum):
    """Durable outcome of one stream-specific live micro-batch."""

    WRITTEN = "written"
    ALREADY_REGISTERED = "already_registered"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class MicroBatchPolicy:
    """Memory and latency bounds for stream-specific micro-batches."""

    max_records: int = 10_000
    max_interval: timedelta = timedelta(seconds=5)
    queue_maxsize: int = 20_000
    stop_poll_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")
        if self.max_interval <= timedelta(0):
            raise ValueError("max_interval must be positive")
        if self.queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be positive")
        interval = float(self.stop_poll_interval_seconds)
        if interval <= 0.0:
            raise ValueError("stop_poll_interval_seconds must be positive")
        object.__setattr__(self, "stop_poll_interval_seconds", interval)


@dataclass(frozen=True, slots=True)
class GapRecoveryPolicy:
    """Historical recovery window calculated from durable checkpoints."""

    enabled: bool = False
    safety_lag: timedelta = timedelta(seconds=2)
    min_gap: timedelta = timedelta(milliseconds=1)
    chunk_size: timedelta | None = timedelta(days=1)
    initial_lookback: timedelta | None = None

    def __post_init__(self) -> None:
        if self.safety_lag < timedelta(0):
            raise ValueError("safety_lag cannot be negative")
        if self.min_gap < timedelta(0):
            raise ValueError("min_gap cannot be negative")
        if self.chunk_size is not None and self.chunk_size <= timedelta(0):
            raise ValueError("chunk_size must be positive")
        if self.initial_lookback is not None and self.initial_lookback <= timedelta(0):
            raise ValueError("initial_lookback must be positive")


@dataclass(frozen=True, slots=True)
class LiveIngestionConfig:
    """Runtime policy for one live ingestion service."""

    pipeline_id: str
    source: str
    schema_version: int = 1
    micro_batch: MicroBatchPolicy = field(default_factory=MicroBatchPolicy)
    reconnect: ReconnectPolicy = field(default_factory=ReconnectPolicy)
    gap_recovery: GapRecoveryPolicy = field(default_factory=GapRecoveryPolicy)
    quality_rejection_policy: QualityRejectionPolicy = QualityRejectionPolicy.RAISE
    quarantine_sample_size: int = 10
    flush_on_shutdown: bool = True
    persist_incomplete_bars: bool = False
    restart_on_stream_end: bool = True

    def __post_init__(self) -> None:
        pipeline_id = self.pipeline_id.strip()
        source = self.source.strip()
        if not pipeline_id:
            raise ValueError("pipeline_id cannot be empty")
        if not source:
            raise ValueError("source cannot be empty")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.quarantine_sample_size < 0:
            raise ValueError("quarantine_sample_size must be non-negative")
        object.__setattr__(self, "pipeline_id", pipeline_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "quality_rejection_policy",
            QualityRejectionPolicy(self.quality_rejection_policy),
        )


@dataclass(frozen=True, slots=True)
class LiveBatchResult:
    """Auditable outcome of persisting one stream-specific micro-batch."""

    key: DatasetKey
    disposition: LiveBatchDisposition
    fetched_records: int
    cleaned_records: int
    cleaning_report: CleaningReport
    quality: QualityDecision
    batch: DatasetBatch | None = None
    checkpoint: IngestionCheckpoint | None = None
    files_created: tuple[Path, ...] = ()
    content_hash: str | None = None
    quarantine_file: Path | None = None


@dataclass(frozen=True, slots=True)
class LiveIngestionResult:
    """Final metrics from one service run."""

    started_at: datetime
    stopped_at: datetime
    received_records: int
    accepted_records: int
    dropped_incomplete_bars: int
    reconnects: int
    recovery_runs: int
    recovery_chunks: int
    batches: tuple[LiveBatchResult, ...]
    transient_errors: tuple[str, ...]

    @property
    def written_batches(self) -> int:
        return sum(item.disposition is LiveBatchDisposition.WRITTEN for item in self.batches)

    @property
    def reused_batches(self) -> int:
        return sum(
            item.disposition is LiveBatchDisposition.ALREADY_REGISTERED
            for item in self.batches
        )

    @property
    def quarantined_batches(self) -> int:
        return sum(
            item.disposition is LiveBatchDisposition.QUARANTINED for item in self.batches
        )

    @property
    def files_created(self) -> tuple[Path, ...]:
        return tuple(path for batch in self.batches for path in batch.files_created)


class MarketDataStore(Protocol):
    def append(self, records: Iterable[MarketDataRecord]) -> StorageWriteResult: ...


class CatalogStore(Protocol):
    def get_by_hash(self, content_hash: str) -> DatasetBatch | None: ...

    def register_batch(
        self,
        registration: DatasetRegistration,
        *,
        batch_id: str | None = None,
    ) -> DatasetBatch: ...


class CheckpointBackend(Protocol):
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
    def write(self, entry: QuarantineEntry) -> Path: ...


class HistoricalRecoveryBackend(Protocol):
    def run(self, plan: HistoricalIngestionPlan) -> HistoricalIngestionResult: ...


@dataclass(slots=True)
class _Buffer:
    opened_at: float
    records: list[MarketDataRecord]


@dataclass(slots=True)
class _MutableStats:
    started_at: datetime
    received_records: int = 0
    accepted_records: int = 0
    dropped_incomplete_bars: int = 0
    reconnects: int = 0
    recovery_runs: int = 0
    recovery_chunks: int = 0
    batches: list[LiveBatchResult] = field(default_factory=list)
    transient_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: Exception


_STREAM_END = object()
_QueueItem: TypeAlias = MarketDataRecord | _StreamFailure | object


class LiveBatchPersister:
    """Synchronously clean and durably commit one homogeneous live micro-batch."""

    def __init__(
        self,
        *,
        config: LiveIngestionConfig,
        cleaner: MarketDataCleaner,
        quality_gate: DataQualityGate,
        storage: MarketDataStore,
        catalog: CatalogStore,
        checkpoints: CheckpointBackend,
        quarantine: QuarantineBackend | None = None,
    ) -> None:
        self.config = config
        self.cleaner = cleaner
        self.quality_gate = quality_gate
        self.storage = storage
        self.catalog = catalog
        self.checkpoints = checkpoints
        self.quarantine = quarantine

    def persist(self, records: Iterable[MarketDataRecord]) -> LiveBatchResult:
        """Persist one stream-specific batch with idempotency and quality checks."""

        fetched = tuple(records)
        if not fetched:
            raise ValueError("records cannot be empty")
        key = _record_key(fetched[0])
        if any(_record_key(record) != key for record in fetched[1:]):
            raise ValueError("A live micro-batch must contain exactly one data stream")

        cleaned = self._clean(fetched, key.kind)
        diagnostics = AdapterDiagnostics(
            rows_read=len(fetched),
            records_emitted=len(fetched),
        )
        decision = self.quality_gate.evaluate(cleaned.report, diagnostics)
        if not decision.accepted:
            quarantine_file = self._quarantine(
                key,
                fetched=fetched,
                report=cleaned.report,
                decision=decision,
                diagnostics=diagnostics,
            )
            if self.config.quality_rejection_policy is QualityRejectionPolicy.RAISE:
                raise LiveIngestionQualityError(decision, quarantine_file)
            return LiveBatchResult(
                key=key,
                disposition=LiveBatchDisposition.QUARANTINED,
                fetched_records=len(fetched),
                cleaned_records=len(cleaned.records),
                cleaning_report=cleaned.report,
                quality=decision,
                quarantine_file=quarantine_file,
            )

        normalized = cast(tuple[MarketDataRecord, ...], cleaned.records)
        if not normalized:
            raise LivePersistenceError("Quality accepted an empty live batch")
        content_hash = compute_content_hash(
            normalized,
            source=self.config.source,
            schema_version=self.config.schema_version,
        )
        start_time, end_time = _coverage(normalized)
        checkpoint_time, checkpoint_sequence = _checkpoint_position(normalized)

        existing = self.catalog.get_by_hash(content_hash)
        if existing is not None:
            self._verify_existing(
                existing,
                key=key,
                start_time=start_time,
                end_time=end_time,
                count=len(normalized),
            )
            checkpoint = self._advance_checkpoint(
                key,
                batch=existing,
                event_time=checkpoint_time,
                sequence=checkpoint_sequence,
            )
            return LiveBatchResult(
                key=key,
                disposition=LiveBatchDisposition.ALREADY_REGISTERED,
                fetched_records=len(fetched),
                cleaned_records=len(normalized),
                cleaning_report=cleaned.report,
                quality=decision,
                batch=existing,
                checkpoint=checkpoint,
                content_hash=content_hash,
            )

        write_result = self.storage.append(normalized)
        self._validate_write_result(write_result, len(normalized))
        registration = DatasetRegistration(
            key=key,
            start_time=start_time,
            end_time=end_time,
            record_count=len(normalized),
            source=self.config.source,
            schema_version=self.config.schema_version,
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
                    count=len(normalized),
                )
            except Exception:
                self._remove_files(write_result.files_created)
                raise
            self._remove_files(write_result.files_created)
            batch = raced
            disposition = LiveBatchDisposition.ALREADY_REGISTERED
            created_files: tuple[Path, ...] = ()
        except Exception:
            self._remove_files(write_result.files_created)
            raise
        else:
            disposition = LiveBatchDisposition.WRITTEN
            created_files = write_result.files_created

        checkpoint = self._advance_checkpoint(
            key,
            batch=batch,
            event_time=checkpoint_time,
            sequence=checkpoint_sequence,
        )
        return LiveBatchResult(
            key=key,
            disposition=disposition,
            fetched_records=len(fetched),
            cleaned_records=len(normalized),
            cleaning_report=cleaned.report,
            quality=decision,
            batch=batch,
            checkpoint=checkpoint,
            files_created=created_files,
            content_hash=content_hash,
        )

    def _clean(
        self,
        records: tuple[MarketDataRecord, ...],
        kind: DataKind,
    ) -> CleanedBatch[Tick] | CleanedBatch[Bar]:
        if kind is DataKind.TICK:
            return self.cleaner.clean_ticks(
                tuple(record for record in records if isinstance(record, Tick))
            )
        return self.cleaner.clean_bars(
            tuple(record for record in records if isinstance(record, Bar))
        )

    def _advance_checkpoint(
        self,
        key: DatasetKey,
        *,
        batch: DatasetBatch,
        event_time: datetime,
        sequence: int | None,
    ) -> IngestionCheckpoint:
        checkpoint_key = CheckpointKey(
            pipeline_id=self.config.pipeline_id,
            source=self.config.source,
            symbol=key.symbol,
            kind=key.kind,
            timeframe=key.timeframe,
        )
        current = self.checkpoints.get(checkpoint_key)
        if current is not None and _position(current.last_event_time, current.last_sequence) >= _position(
            event_time, sequence
        ):
            return current
        try:
            return self.checkpoints.advance(
                checkpoint_key,
                last_event_time=event_time,
                last_sequence=sequence,
                last_batch_id=batch.batch_id,
            )
        except Exception as exc:
            raise LiveCheckpointError(batch, exc) from exc

    def _quarantine(
        self,
        key: DatasetKey,
        *,
        fetched: tuple[MarketDataRecord, ...],
        report: CleaningReport,
        decision: QualityDecision,
        diagnostics: AdapterDiagnostics,
    ) -> Path | None:
        if self.quarantine is None:
            return None
        start, end = _coverage(fetched)
        sample = tuple(
            _sample_payload(record)
            for record in fetched[: self.config.quarantine_sample_size]
        )
        return self.quarantine.write(
            QuarantineEntry(
                pipeline_id=self.config.pipeline_id,
                source=self.config.source,
                symbol=key.symbol,
                kind=key.kind,
                timeframe=key.timeframe,
                start_time=start,
                end_time=end,
                fetched_records=len(fetched),
                diagnostics=diagnostics,
                cleaning_report=report,
                decision=decision,
                sample_records=sample,
            )
        )

    def _verify_existing(
        self,
        batch: DatasetBatch,
        *,
        key: DatasetKey,
        start_time: datetime,
        end_time: datetime,
        count: int,
    ) -> None:
        compatible = (
            batch.key == key
            and batch.start_time == start_time
            and batch.end_time == end_time
            and batch.record_count == count
            and batch.source == self.config.source
            and batch.schema_version == self.config.schema_version
        )
        if not compatible:
            raise IngestionConflictError(
                f"Content hash {batch.content_hash!r} has incompatible live metadata"
            )

    @staticmethod
    def _validate_write_result(result: StorageWriteResult, expected: int) -> None:
        if result.records_written != expected:
            LiveBatchPersister._remove_files(result.files_created)
            raise LivePersistenceError(
                f"Storage reported {result.records_written} records, expected {expected}"
            )
        if not result.files_created:
            raise LivePersistenceError("Storage did not create any durable files")

    @staticmethod
    def _remove_files(files: Iterable[Path]) -> None:
        for path in files:
            with suppress(OSError):
                Path(path).unlink(missing_ok=True)


class LiveIngestionService:
    """Run live ingestion until stopped, reconnecting and recovering gaps safely."""

    def __init__(
        self,
        *,
        adapter: LiveMarketDataAdapter,
        persister: LiveBatchPersister,
        config: LiveIngestionConfig,
        recovery: HistoricalRecoveryBackend | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if config.gap_recovery.enabled and recovery is None:
            raise LiveIngestionConfigurationError(
                "gap recovery is enabled but no historical coordinator was supplied"
            )
        if persister.config != config:
            raise LiveIngestionConfigurationError(
                "LiveBatchPersister and LiveIngestionService must share one config"
            )
        self.adapter = adapter
        self.persister = persister
        self.config = config
        self.recovery = recovery
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self._random_value = random_value
        self._stop_event = asyncio.Event()
        self._running = False
        self._buffers: dict[DatasetKey, _Buffer] = {}
        self._stats: _MutableStats | None = None

    def request_stop(self) -> None:
        """Request graceful shutdown; buffered records are flushed by default."""

        self._stop_event.set()

    async def run(
        self,
        subscription: LiveSubscription,
        *,
        stop_event: asyncio.Event | None = None,
    ) -> LiveIngestionResult:
        """Connect, recover gaps, stream records, and return final run metrics."""

        if self._running:
            raise RuntimeError("LiveIngestionService is already running")
        self._running = True
        self._stop_event = asyncio.Event()
        self._buffers.clear()
        self._stats = _MutableStats(started_at=_utc_now(self._now()))
        stats = self._require_stats()
        retries = 0
        try:
            while not self._stopping(stop_event):
                session_received_before = stats.received_records
                try:
                    await self.adapter.connect()
                    await self._recover_gaps(subscription)
                    await self._consume_session(subscription, stop_event)
                    if self._stopping(stop_event):
                        break
                    if not self.config.restart_on_stream_end:
                        break
                    raise LiveStreamEndedError("Live adapter stream ended unexpectedly")
                except LiveIngestionError:
                    raise
                except asyncio.CancelledError:
                    self.request_stop()
                    raise
                except Exception as exc:
                    await self._flush_all()
                    await self._safe_disconnect()
                    if stats.received_records > session_received_before:
                        retries = 0
                    if not self.config.reconnect.allows_retry(retries):
                        raise LiveReconnectExhaustedError(retries, exc) from exc
                    retries += 1
                    stats.reconnects += 1
                    stats.transient_errors.append(f"{type(exc).__name__}: {exc}")
                    delay = self.config.reconnect.delay_seconds(
                        retries,
                        random_value=self._random_value(),
                    )
                    await self._sleep(delay)
                else:
                    retries = 0

            if self.config.flush_on_shutdown:
                await self._flush_all()
            return self._result()
        finally:
            await self._safe_disconnect()
            self._running = False

    async def flush(self) -> tuple[LiveBatchResult, ...]:
        """Flush every currently buffered stream and return new batch results."""

        before = len(self._require_stats().batches)
        await self._flush_all()
        return tuple(self._require_stats().batches[before:])

    async def _consume_session(
        self,
        subscription: LiveSubscription,
        external_stop: asyncio.Event | None,
    ) -> None:
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue(
            maxsize=self.config.micro_batch.queue_maxsize
        )
        producer = asyncio.create_task(self._pump(subscription, queue))
        try:
            while not self._stopping(external_stop):
                timeout = self._next_wait_timeout()
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                except TimeoutError:
                    await self._flush_due()
                    continue

                if item is _STREAM_END:
                    return
                if isinstance(item, _StreamFailure):
                    raise item.error
                record = cast(MarketDataRecord, item)
                stats = self._require_stats()
                stats.received_records += 1
                if isinstance(record, Bar) and not record.complete and not self.config.persist_incomplete_bars:
                    stats.dropped_incomplete_bars += 1
                    continue
                stats.accepted_records += 1
                key = _record_key(record)
                buffer = self._buffers.get(key)
                if buffer is None:
                    buffer = _Buffer(opened_at=self._monotonic(), records=[])
                    self._buffers[key] = buffer
                buffer.records.append(record)
                if len(buffer.records) >= self.config.micro_batch.max_records:
                    await self._flush_key(key)
                await self._flush_due()
        finally:
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer

    async def _pump(
        self,
        subscription: LiveSubscription,
        queue: asyncio.Queue[_QueueItem],
    ) -> None:
        try:
            async for record in self.adapter.stream(subscription):
                await queue.put(record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(_StreamFailure(exc))
        else:
            await queue.put(_STREAM_END)

    async def _flush_due(self) -> None:
        now = self._monotonic()
        maximum_age = self.config.micro_batch.max_interval.total_seconds()
        due = [
            key
            for key, buffer in self._buffers.items()
            if buffer.records and now - buffer.opened_at >= maximum_age
        ]
        for key in sorted(due, key=_key_sort):
            await self._flush_key(key)

    async def _flush_all(self) -> None:
        for key in sorted(tuple(self._buffers), key=_key_sort):
            await self._flush_key(key)

    async def _flush_key(self, key: DatasetKey) -> None:
        buffer = self._buffers.get(key)
        if buffer is None or not buffer.records:
            self._buffers.pop(key, None)
            return
        records = tuple(buffer.records)
        result = await asyncio.to_thread(self.persister.persist, records)
        self._buffers.pop(key, None)
        self._require_stats().batches.append(result)

    def _next_wait_timeout(self) -> float:
        poll = self.config.micro_batch.stop_poll_interval_seconds
        if not self._buffers:
            return poll
        now = self._monotonic()
        interval = self.config.micro_batch.max_interval.total_seconds()
        remaining = min(
            max(0.0, buffer.opened_at + interval - now)
            for buffer in self._buffers.values()
            if buffer.records
        )
        return min(poll, remaining)

    async def _recover_gaps(self, subscription: LiveSubscription) -> None:
        policy = self.config.gap_recovery
        if not policy.enabled:
            return
        recovery = cast(HistoricalRecoveryBackend, self.recovery)
        recovery_end = _utc_now(self._now()) - policy.safety_lag
        for symbol in sorted(subscription.symbols):
            for timeframe in sorted(
                subscription.timeframes,
                key=lambda item: (item.seconds is not None, item.seconds or 0),
            ):
                kind = DataKind.TICK if timeframe is Timeframe.TICK else DataKind.BAR
                checkpoint_key = CheckpointKey(
                    pipeline_id=self.config.pipeline_id,
                    source=self.config.source,
                    symbol=symbol,
                    kind=kind,
                    timeframe=None if kind is DataKind.TICK else timeframe,
                )
                current = self.persister.checkpoints.get(checkpoint_key)
                start = _recovery_start(current, timeframe, recovery_end, policy)
                if start is None or recovery_end <= start:
                    continue
                if recovery_end - start < policy.min_gap:
                    continue
                request = HistoricalDataRequest(
                    symbol=symbol,
                    kind=kind,
                    timeframe=None if kind is DataKind.TICK else timeframe,
                    start=start,
                    end=recovery_end,
                )
                plan = HistoricalIngestionPlan(
                    pipeline_id=self.config.pipeline_id,
                    source=self.config.source,
                    request=request,
                    chunk_size=policy.chunk_size,
                    schema_version=self.config.schema_version,
                    quality_rejection_policy=self.config.quality_rejection_policy,
                    quarantine_sample_size=self.config.quarantine_sample_size,
                )
                try:
                    result = await asyncio.to_thread(recovery.run, plan)
                except MarketDataAdapterError:
                    raise
                except Exception as exc:
                    raise LiveGapRecoveryError(
                        f"Gap recovery failed for {symbol}/{timeframe.value}: {exc}"
                    ) from exc
                stats = self._require_stats()
                stats.recovery_runs += 1
                stats.recovery_chunks += len(result.chunks)

    def _stopping(self, external: asyncio.Event | None) -> bool:
        return self._stop_event.is_set() or (external is not None and external.is_set())

    async def _safe_disconnect(self) -> None:
        with suppress(Exception):
            await self.adapter.disconnect()

    def _result(self) -> LiveIngestionResult:
        stats = self._require_stats()
        return LiveIngestionResult(
            started_at=stats.started_at,
            stopped_at=_utc_now(self._now()),
            received_records=stats.received_records,
            accepted_records=stats.accepted_records,
            dropped_incomplete_bars=stats.dropped_incomplete_bars,
            reconnects=stats.reconnects,
            recovery_runs=stats.recovery_runs,
            recovery_chunks=stats.recovery_chunks,
            batches=tuple(stats.batches),
            transient_errors=tuple(stats.transient_errors),
        )

    def _require_stats(self) -> _MutableStats:
        if self._stats is None:
            raise RuntimeError("LiveIngestionService has not been started")
        return self._stats


def _record_key(record: MarketDataRecord) -> DatasetKey:
    if isinstance(record, Tick):
        return DatasetKey(DataKind.TICK, record.symbol)
    return DatasetKey(DataKind.BAR, record.symbol, record.timeframe)


def _record_time(record: MarketDataRecord) -> datetime:
    return record.event_time if isinstance(record, Tick) else record.open_time


def _coverage(records: tuple[MarketDataRecord, ...]) -> tuple[datetime, datetime]:
    if isinstance(records[0], Tick):
        ticks = tuple(record for record in records if isinstance(record, Tick))
        start = min(record.event_time for record in ticks)
        end = max(record.event_time for record in ticks) + timedelta(microseconds=1)
        return start, end
    bars = tuple(record for record in records if isinstance(record, Bar))
    return min(record.open_time for record in bars), max(record.close_time for record in bars)


def _checkpoint_position(
    records: tuple[MarketDataRecord, ...],
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


def _position(event_time: datetime, sequence: int | None) -> tuple[datetime, int]:
    return event_time, sequence if sequence is not None else -1


def _key_sort(key: DatasetKey) -> tuple[str, str, str]:
    return key.kind.value, key.symbol, key.timeframe.value if key.timeframe is not None else ""


def _recovery_start(
    checkpoint: IngestionCheckpoint | None,
    timeframe: Timeframe,
    recovery_end: datetime,
    policy: GapRecoveryPolicy,
) -> datetime | None:
    if checkpoint is None:
        if policy.initial_lookback is None:
            return None
        return recovery_end - policy.initial_lookback
    if timeframe is Timeframe.TICK:
        return checkpoint.last_event_time + timedelta(microseconds=1)
    seconds = timeframe.seconds
    if seconds is None:  # pragma: no cover - defensive
        raise RuntimeError("Bar timeframe requires a fixed interval")
    return checkpoint.last_event_time + timedelta(seconds=seconds)


def _sample_payload(record: MarketDataRecord) -> dict[str, object]:
    if isinstance(record, Tick):
        return {
            "kind": "tick",
            "symbol": record.symbol,
            "event_time": record.event_time.isoformat(timespec="microseconds"),
            "bid": record.bid,
            "ask": record.ask,
            "sequence": record.sequence,
        }
    return {
        "kind": "bar",
        "symbol": record.symbol,
        "open_time": record.open_time.isoformat(timespec="microseconds"),
        "timeframe": record.timeframe.value,
        "complete": record.complete,
    }


def _utc_now(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now provider must return a timezone-aware datetime")
    return value.astimezone(UTC)
