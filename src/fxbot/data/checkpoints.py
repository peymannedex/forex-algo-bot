"""Transactional SQLite checkpoints for resumable ingestion pipelines."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fxbot.domain.enums import DataKind, Timeframe

_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for checkpoint-store failures."""


class CheckpointRegressionError(CheckpointError):
    """Raised when a checkpoint attempts to move behind its committed position."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a persisted checkpoint cannot be reconstructed safely."""


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat(timespec="microseconds")


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise CheckpointCorruptionError(f"{field_name} must be stored as text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CheckpointCorruptionError(f"Invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CheckpointCorruptionError(f"Stored {field_name} is not timezone-aware")
    return parsed.astimezone(UTC)


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _symbol(value: str) -> str:
    return _non_empty(value, "symbol").upper()


def _timeframe(kind: DataKind, value: Timeframe | str | None) -> Timeframe | None:
    if kind is DataKind.TICK:
        if value is not None and Timeframe.parse(value) is not Timeframe.TICK:
            raise ValueError("Tick checkpoints cannot specify a bar timeframe")
        return None
    if value is None:
        raise ValueError("Bar checkpoints require a non-tick timeframe")
    parsed = Timeframe.parse(value)
    if parsed is Timeframe.TICK:
        raise ValueError("Bar checkpoints require a non-tick timeframe")
    return parsed


@dataclass(frozen=True, slots=True)
class CheckpointKey:
    """Identity of one independently resumable ingestion stream."""

    pipeline_id: str
    source: str
    symbol: str
    kind: DataKind
    timeframe: Timeframe | None = None

    def __post_init__(self) -> None:
        kind = DataKind(self.kind)
        object.__setattr__(self, "pipeline_id", _non_empty(self.pipeline_id, "pipeline_id"))
        object.__setattr__(self, "source", _non_empty(self.source, "source"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "timeframe", _timeframe(kind, self.timeframe))


@dataclass(frozen=True, slots=True)
class IngestionCheckpoint:
    """Last fully committed event position for an ingestion stream."""

    key: CheckpointKey
    last_event_time: datetime
    last_sequence: int | None = None
    last_batch_id: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        event_time = _utc(self.last_event_time, "last_event_time")
        updated = _utc(self.updated_at, "updated_at")
        if self.last_sequence is not None and self.last_sequence < 0:
            raise ValueError("last_sequence must be non-negative")
        batch_id = (
            _non_empty(self.last_batch_id, "last_batch_id")
            if self.last_batch_id is not None
            else None
        )
        object.__setattr__(self, "last_event_time", event_time)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "last_batch_id", batch_id)


class CheckpointStore:
    """SQLite store that atomically advances monotonic ingestion checkpoints."""

    def __init__(self, database: str | Path, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.database = Path(database).expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get(self, key: CheckpointKey) -> IngestionCheckpoint | None:
        """Return the current checkpoint for one stream."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM ingestion_checkpoints
                WHERE pipeline_id = ? AND source = ? AND symbol = ?
                  AND kind = ? AND timeframe_key = ?
                """,
                self._key_parameters(key),
            ).fetchone()
        return self._row_to_checkpoint(row) if row is not None else None

    def advance(
        self,
        key: CheckpointKey,
        *,
        last_event_time: datetime,
        last_sequence: int | None = None,
        last_batch_id: str | None = None,
        updated_at: datetime | None = None,
        allow_regression: bool = False,
    ) -> IngestionCheckpoint:
        """Atomically create or advance a checkpoint.

        Positions are ordered by ``(last_event_time, last_sequence)``.  At equal
        timestamps, a known sequence is later than an unknown sequence.  A
        backwards move raises :class:`CheckpointRegressionError` unless an
        explicit recovery operation sets ``allow_regression=True``.
        """

        proposed = IngestionCheckpoint(
            key=key,
            last_event_time=last_event_time,
            last_sequence=last_sequence,
            last_batch_id=last_batch_id,
            updated_at=updated_at if updated_at is not None else datetime.now(UTC),
        )

        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM ingestion_checkpoints
                WHERE pipeline_id = ? AND source = ? AND symbol = ?
                  AND kind = ? AND timeframe_key = ?
                """,
                self._key_parameters(key),
            ).fetchone()
            current = self._row_to_checkpoint(row) if row is not None else None
            if current is not None and not allow_regression and self._position(proposed) < self._position(current):
                raise CheckpointRegressionError(
                    f"Checkpoint regression for {key}: "
                    f"{self._position(proposed)} < {self._position(current)}"
                )

            connection.execute(
                """
                INSERT INTO ingestion_checkpoints (
                    pipeline_id, source, symbol, kind, timeframe, timeframe_key,
                    last_event_time, last_sequence, last_batch_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pipeline_id, source, symbol, kind, timeframe_key)
                DO UPDATE SET
                    last_event_time = excluded.last_event_time,
                    last_sequence = excluded.last_sequence,
                    last_batch_id = excluded.last_batch_id,
                    updated_at = excluded.updated_at
                """,
                (
                    key.pipeline_id,
                    key.source,
                    key.symbol,
                    key.kind.value,
                    key.timeframe.value if key.timeframe is not None else None,
                    key.timeframe.value if key.timeframe is not None else "",
                    _timestamp(proposed.last_event_time),
                    proposed.last_sequence,
                    proposed.last_batch_id,
                    _timestamp(proposed.updated_at),
                ),
            )

        stored = self.get(key)
        if stored is None:  # pragma: no cover - defensive postcondition
            raise CheckpointError(f"Checkpoint for {key} could not be reloaded")
        return stored

    def list(
        self,
        *,
        pipeline_id: str | None = None,
        source: str | None = None,
    ) -> tuple[IngestionCheckpoint, ...]:
        """List checkpoints, optionally filtered by pipeline and source."""

        clauses: list[str] = []
        parameters: list[object] = []
        if pipeline_id is not None:
            clauses.append("pipeline_id = ?")
            parameters.append(_non_empty(pipeline_id, "pipeline_id"))
        if source is not None:
            clauses.append("source = ?")
            parameters.append(_non_empty(source, "source"))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT * FROM ingestion_checkpoints"
            f"{where} ORDER BY pipeline_id, source, symbol, kind, timeframe_key"
        )
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._row_to_checkpoint(row) for row in rows)

    def delete(self, key: CheckpointKey) -> bool:
        """Delete one checkpoint."""

        with self._transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM ingestion_checkpoints
                WHERE pipeline_id = ? AND source = ? AND symbol = ?
                  AND kind = ? AND timeframe_key = ?
                """,
                self._key_parameters(key),
            )
        return cursor.rowcount == 1

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
                    pipeline_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('tick', 'bar')),
                    timeframe TEXT,
                    timeframe_key TEXT NOT NULL,
                    last_event_time TEXT NOT NULL,
                    last_sequence INTEGER CHECK (last_sequence IS NULL OR last_sequence >= 0),
                    last_batch_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (pipeline_id, source, symbol, kind, timeframe_key),
                    CHECK (
                        (kind = 'tick' AND timeframe IS NULL)
                        OR
                        (kind = 'bar' AND timeframe IS NOT NULL AND timeframe <> 'tick')
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoint_pipeline_source
                ON ingestion_checkpoints(pipeline_id, source);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO checkpoint_meta(key, value) VALUES ('schema_version', ?)",
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
    def _key_parameters(key: CheckpointKey) -> tuple[object, ...]:
        return (
            key.pipeline_id,
            key.source,
            key.symbol,
            key.kind.value,
            key.timeframe.value if key.timeframe is not None else "",
        )

    @staticmethod
    def _position(checkpoint: IngestionCheckpoint) -> tuple[datetime, int]:
        sequence = checkpoint.last_sequence if checkpoint.last_sequence is not None else -1
        return checkpoint.last_event_time, sequence

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> IngestionCheckpoint:
        try:
            kind = DataKind(str(row["kind"]))
            raw_timeframe = row["timeframe"]
            timeframe = Timeframe.parse(str(raw_timeframe)) if raw_timeframe is not None else None
            raw_sequence = row["last_sequence"]
            sequence = int(raw_sequence) if raw_sequence is not None else None
            raw_batch_id = row["last_batch_id"]
            batch_id = str(raw_batch_id) if raw_batch_id is not None else None
            return IngestionCheckpoint(
                key=CheckpointKey(
                    pipeline_id=str(row["pipeline_id"]),
                    source=str(row["source"]),
                    symbol=str(row["symbol"]),
                    kind=kind,
                    timeframe=timeframe,
                ),
                last_event_time=_parse_timestamp(row["last_event_time"], "last_event_time"),
                last_sequence=sequence,
                last_batch_id=batch_id,
                updated_at=_parse_timestamp(row["updated_at"], "updated_at"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptionError(f"Cannot reconstruct checkpoint: {exc}") from exc
