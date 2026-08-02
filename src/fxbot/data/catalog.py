"""SQLite-backed catalog for persisted market-data batches.

The catalog stores metadata only; canonical tick and bar payloads remain in the
Parquet partition store.  Every write is transactional and content hashes are
unique, making historical ingestion retries idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fxbot.domain.enums import DataKind, Timeframe

_SCHEMA_VERSION = 1


class CatalogError(RuntimeError):
    """Base class for data-catalog failures."""


class CatalogConflictError(CatalogError):
    """Raised when an idempotency key conflicts with different metadata."""


class CatalogCorruptionError(CatalogError):
    """Raised when a persisted catalog row cannot be reconstructed safely."""


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat(timespec="microseconds")


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise CatalogCorruptionError(f"{field_name} must be stored as text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CatalogCorruptionError(f"Invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CatalogCorruptionError(f"Stored {field_name} is not timezone-aware")
    return parsed.astimezone(UTC)


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _timeframe(kind: DataKind, value: Timeframe | str | None) -> Timeframe | None:
    if kind is DataKind.TICK:
        if value is not None and Timeframe.parse(value) is not Timeframe.TICK:
            raise ValueError("Tick datasets cannot specify a bar timeframe")
        return None
    if value is None:
        raise ValueError("Bar datasets require a non-tick timeframe")
    parsed = Timeframe.parse(value)
    if parsed is Timeframe.TICK:
        raise ValueError("Bar datasets require a non-tick timeframe")
    return parsed


@dataclass(frozen=True, slots=True)
class DatasetKey:
    """Identity of one logical market-data stream."""

    kind: DataKind
    symbol: str
    timeframe: Timeframe | None = None

    def __post_init__(self) -> None:
        kind = DataKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "timeframe", _timeframe(kind, self.timeframe))


@dataclass(frozen=True, slots=True)
class DatasetRegistration:
    """Metadata supplied when a new persisted batch is registered."""

    key: DatasetKey
    start_time: datetime
    end_time: datetime
    record_count: int
    source: str
    schema_version: int
    content_hash: str
    files: tuple[Path, ...]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        start = _utc(self.start_time, "start_time")
        end = _utc(self.end_time, "end_time")
        created = _utc(self.created_at, "created_at")
        if start >= end:
            raise ValueError("start_time must be earlier than end_time")
        if self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        content_hash = _non_empty(self.content_hash, "content_hash").lower()
        source = _non_empty(self.source, "source")
        files = tuple(Path(path) for path in self.files)
        if not files:
            raise ValueError("files cannot be empty")
        if any(not str(path).strip() for path in files):
            raise ValueError("file paths cannot be empty")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class DatasetBatch:
    """Persisted catalog representation of one immutable data batch."""

    batch_id: str
    key: DatasetKey
    start_time: datetime
    end_time: datetime
    record_count: int
    source: str
    schema_version: int
    content_hash: str
    files: tuple[Path, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        batch_id = _non_empty(self.batch_id, "batch_id")
        validated = DatasetRegistration(
            key=self.key,
            start_time=self.start_time,
            end_time=self.end_time,
            record_count=self.record_count,
            source=self.source,
            schema_version=self.schema_version,
            content_hash=self.content_hash,
            files=self.files,
            created_at=self.created_at,
        )
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "start_time", validated.start_time)
        object.__setattr__(self, "end_time", validated.end_time)
        object.__setattr__(self, "source", validated.source)
        object.__setattr__(self, "content_hash", validated.content_hash)
        object.__setattr__(self, "files", validated.files)
        object.__setattr__(self, "created_at", validated.created_at)


@dataclass(frozen=True, slots=True)
class CoverageInterval:
    """Merged half-open coverage interval: ``start <= t < end``."""

    start_time: datetime
    end_time: datetime

    def __post_init__(self) -> None:
        start = _utc(self.start_time, "start_time")
        end = _utc(self.end_time, "end_time")
        if start >= end:
            raise ValueError("start_time must be earlier than end_time")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)

    def contains(self, timestamp: datetime) -> bool:
        value = _utc(timestamp, "timestamp")
        return self.start_time <= value < self.end_time


class DataCatalog:
    """Transactional SQLite metadata catalog for immutable market-data batches."""

    def __init__(self, database: str | Path, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.database = Path(database).expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_batch(
        self,
        registration: DatasetRegistration,
        *,
        batch_id: str | None = None,
    ) -> DatasetBatch:
        """Register a batch or return the identical existing hash registration.

        The content hash is the idempotency key.  Reusing it with different
        metadata raises :class:`CatalogConflictError` rather than silently
        accepting inconsistent provenance.
        """

        requested_id = _non_empty(batch_id, "batch_id") if batch_id is not None else uuid4().hex
        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM dataset_batches WHERE content_hash = ?",
                (registration.content_hash,),
            ).fetchone()
            if existing_row is not None:
                existing = self._row_to_batch(existing_row)
                if self._matches(existing, registration):
                    return existing
                raise CatalogConflictError(
                    f"content_hash {registration.content_hash!r} is already registered "
                    "with different metadata"
                )

            try:
                connection.execute(
                    """
                    INSERT INTO dataset_batches (
                        batch_id, kind, symbol, timeframe, start_time, end_time,
                        record_count, source, schema_version, content_hash,
                        files_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        requested_id,
                        registration.key.kind.value,
                        registration.key.symbol,
                        registration.key.timeframe.value
                        if registration.key.timeframe is not None
                        else None,
                        _timestamp(registration.start_time),
                        _timestamp(registration.end_time),
                        registration.record_count,
                        registration.source,
                        registration.schema_version,
                        registration.content_hash,
                        json.dumps([str(path) for path in registration.files]),
                        _timestamp(registration.created_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CatalogConflictError(f"Cannot register batch {requested_id!r}: {exc}") from exc

        stored = self.get_batch(requested_id)
        if stored is None:  # pragma: no cover - defensive postcondition
            raise CatalogError(f"Registered batch {requested_id!r} could not be reloaded")
        return stored

    def get_batch(self, batch_id: str) -> DatasetBatch | None:
        """Return one batch by identifier."""

        normalized = _non_empty(batch_id, "batch_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_batches WHERE batch_id = ?", (normalized,)
            ).fetchone()
        return self._row_to_batch(row) if row is not None else None

    def get_by_hash(self, content_hash: str) -> DatasetBatch | None:
        """Return one batch by its idempotency hash."""

        normalized = _non_empty(content_hash, "content_hash").lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_batches WHERE content_hash = ?", (normalized,)
            ).fetchone()
        return self._row_to_batch(row) if row is not None else None

    def contains_hash(self, content_hash: str) -> bool:
        """Return whether a content hash is already registered."""

        return self.get_by_hash(content_hash) is not None

    def list_batches(
        self,
        *,
        key: DatasetKey | None = None,
        source: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[DatasetBatch, ...]:
        """List batches overlapping an optional half-open query window."""

        normalized_start = _utc(start, "start") if start is not None else None
        normalized_end = _utc(end, "end") if end is not None else None
        if normalized_start is not None and normalized_end is not None and normalized_start >= normalized_end:
            raise ValueError("start must be earlier than end")

        clauses: list[str] = []
        parameters: list[object] = []
        if key is not None:
            clauses.extend(("kind = ?", "symbol = ?", "timeframe IS ?"))
            parameters.extend(
                (
                    key.kind.value,
                    key.symbol,
                    key.timeframe.value if key.timeframe is not None else None,
                )
            )
        if source is not None:
            clauses.append("source = ?")
            parameters.append(_non_empty(source, "source"))
        if normalized_start is not None:
            clauses.append("end_time > ?")
            parameters.append(_timestamp(normalized_start))
        if normalized_end is not None:
            clauses.append("start_time < ?")
            parameters.append(_timestamp(normalized_end))

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM dataset_batches{where} ORDER BY start_time, end_time, batch_id"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._row_to_batch(row) for row in rows)

    def get_coverage(
        self,
        key: DatasetKey,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[CoverageInterval, ...]:
        """Return merged coverage intervals clipped to an optional query range."""

        normalized_start = _utc(start, "start") if start is not None else None
        normalized_end = _utc(end, "end") if end is not None else None
        if normalized_start is not None and normalized_end is not None and normalized_start >= normalized_end:
            raise ValueError("start must be earlier than end")

        batches = self.list_batches(key=key, start=normalized_start, end=normalized_end)
        intervals: list[CoverageInterval] = []
        for batch in batches:
            interval_start = max(batch.start_time, normalized_start) if normalized_start is not None else batch.start_time
            interval_end = min(batch.end_time, normalized_end) if normalized_end is not None else batch.end_time
            if interval_start >= interval_end:
                continue
            if intervals and interval_start <= intervals[-1].end_time:
                previous = intervals[-1]
                intervals[-1] = CoverageInterval(
                    previous.start_time,
                    max(previous.end_time, interval_end),
                )
            else:
                intervals.append(CoverageInterval(interval_start, interval_end))
        return tuple(intervals)

    def remove_batch(self, batch_id: str) -> bool:
        """Remove catalog metadata for a batch; Parquet files are not deleted."""

        normalized = _non_empty(batch_id, "batch_id")
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM dataset_batches WHERE batch_id = ?", (normalized,)
            )
        return cursor.rowcount == 1

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_batches (
                    batch_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('tick', 'bar')),
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    record_count INTEGER NOT NULL CHECK (record_count > 0),
                    source TEXT NOT NULL,
                    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
                    content_hash TEXT NOT NULL UNIQUE,
                    files_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    CHECK (start_time < end_time),
                    CHECK (
                        (kind = 'tick' AND timeframe IS NULL)
                        OR
                        (kind = 'bar' AND timeframe IS NOT NULL AND timeframe <> 'tick')
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_dataset_stream_time
                ON dataset_batches(kind, symbol, timeframe, start_time, end_time);

                CREATE INDEX IF NOT EXISTS idx_dataset_source
                ON dataset_batches(source);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _matches(batch: DatasetBatch, registration: DatasetRegistration) -> bool:
        return (
            batch.key == registration.key
            and batch.start_time == registration.start_time
            and batch.end_time == registration.end_time
            and batch.record_count == registration.record_count
            and batch.source == registration.source
            and batch.schema_version == registration.schema_version
            and batch.content_hash == registration.content_hash
            and batch.files == registration.files
        )

    @staticmethod
    def _row_to_batch(row: sqlite3.Row) -> DatasetBatch:
        try:
            kind = DataKind(str(row["kind"]))
            raw_timeframe = row["timeframe"]
            timeframe = Timeframe.parse(str(raw_timeframe)) if raw_timeframe is not None else None
            raw_files: Any = json.loads(str(row["files_json"]))
            if not isinstance(raw_files, list) or not all(isinstance(item, str) for item in raw_files):
                raise ValueError("files_json must contain a list of strings")
            return DatasetBatch(
                batch_id=str(row["batch_id"]),
                key=DatasetKey(kind=kind, symbol=str(row["symbol"]), timeframe=timeframe),
                start_time=_parse_timestamp(row["start_time"], "start_time"),
                end_time=_parse_timestamp(row["end_time"], "end_time"),
                record_count=int(row["record_count"]),
                source=str(row["source"]),
                schema_version=int(row["schema_version"]),
                content_hash=str(row["content_hash"]),
                files=tuple(Path(item) for item in raw_files),
                created_at=_parse_timestamp(row["created_at"], "created_at"),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CatalogCorruptionError(f"Cannot reconstruct dataset batch: {exc}") from exc
