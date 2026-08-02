from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.data.checkpoints import (
    CheckpointCorruptionError,
    CheckpointKey,
    CheckpointRegressionError,
    CheckpointStore,
    IngestionCheckpoint,
)
from fxbot.domain.enums import DataKind, Timeframe

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def tick_key(symbol: str = "eurusd") -> CheckpointKey:
    return CheckpointKey("mt5-historical", "mt5", symbol, DataKind.TICK)


def test_checkpoint_store_initializes_reopens_and_preserves_position(tmp_path) -> None:
    database = tmp_path / "state" / "catalog.sqlite3"
    store = CheckpointStore(database)
    saved = store.advance(
        tick_key(),
        last_event_time=BASE,
        last_sequence=7,
        last_batch_id="batch-1",
        updated_at=BASE + timedelta(seconds=1),
    )

    reopened = CheckpointStore(database)

    assert reopened.get(tick_key()) == saved
    assert saved.key.symbol == "EURUSD"
    assert saved.last_sequence == 7
    assert saved.last_batch_id == "batch-1"


def test_checkpoint_keys_enforce_stream_timeframe_rules() -> None:
    assert tick_key().timeframe is None
    bar = CheckpointKey("pipeline", "mt5", "eurusd", DataKind.BAR, Timeframe.M15)
    assert bar.timeframe is Timeframe.M15

    with pytest.raises(ValueError, match="Tick checkpoints"):
        CheckpointKey("pipeline", "mt5", "EURUSD", DataKind.TICK, Timeframe.M1)
    with pytest.raises(ValueError, match="Bar checkpoints"):
        CheckpointKey("pipeline", "mt5", "EURUSD", DataKind.BAR)
    with pytest.raises(ValueError, match="cannot be empty"):
        CheckpointKey("", "mt5", "EURUSD", DataKind.TICK)


def test_checkpoint_model_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        IngestionCheckpoint(tick_key(), datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="last_sequence"):
        IngestionCheckpoint(tick_key(), BASE, last_sequence=-1)
    with pytest.raises(ValueError, match="last_batch_id"):
        IngestionCheckpoint(tick_key(), BASE, last_batch_id="  ")


def test_advance_is_monotonic_by_time_and_sequence(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints.sqlite3")
    key = tick_key()
    first = store.advance(key, last_event_time=BASE, last_sequence=10)
    later_sequence = store.advance(key, last_event_time=BASE, last_sequence=11)
    later_time = store.advance(
        key,
        last_event_time=BASE + timedelta(seconds=1),
        last_sequence=5,
    )

    assert first.last_sequence == 10
    assert later_sequence.last_sequence == 11
    assert later_time.last_event_time == BASE + timedelta(seconds=1)

    with pytest.raises(CheckpointRegressionError):
        store.advance(key, last_event_time=BASE, last_sequence=12)
    with pytest.raises(CheckpointRegressionError):
        store.advance(
            key,
            last_event_time=BASE + timedelta(seconds=1),
            last_sequence=None,
        )


def test_known_sequence_advances_unknown_at_same_timestamp(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints.sqlite3")
    key = tick_key()
    store.advance(key, last_event_time=BASE)

    advanced = store.advance(key, last_event_time=BASE, last_sequence=0)

    assert advanced.last_sequence == 0


def test_explicit_recovery_can_move_checkpoint_backward(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints.sqlite3")
    key = tick_key()
    store.advance(key, last_event_time=BASE + timedelta(hours=2), last_sequence=5)

    recovered = store.advance(
        key,
        last_event_time=BASE,
        last_sequence=1,
        last_batch_id="recovery-batch",
        allow_regression=True,
    )

    assert recovered.last_event_time == BASE
    assert recovered.last_sequence == 1
    assert recovered.last_batch_id == "recovery-batch"


def test_list_filters_and_delete_are_deterministic(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints.sqlite3")
    tick = tick_key("EURUSD")
    bar = CheckpointKey("mt5-live", "mt5", "GBPUSD", DataKind.BAR, Timeframe.M1)
    other = CheckpointKey("csv-import", "csv", "USDJPY", DataKind.TICK)
    store.advance(tick, last_event_time=BASE)
    store.advance(bar, last_event_time=BASE)
    store.advance(other, last_event_time=BASE)

    assert {item.key for item in store.list(source="mt5")} == {tick, bar}
    assert [item.key for item in store.list(pipeline_id="csv-import")] == [other]
    assert store.delete(tick)
    assert not store.delete(tick)
    assert store.get(tick) is None


def test_checkpoint_wal_allows_write_with_open_reader(tmp_path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    store = CheckpointStore(database)
    store.advance(tick_key(), last_event_time=BASE)

    reader = sqlite3.connect(database, isolation_level=None)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM ingestion_checkpoints").fetchone() == (1,)
        saved = store.advance(
            tick_key("GBPUSD"),
            last_event_time=BASE + timedelta(seconds=1),
        )
        assert saved.key.symbol == "GBPUSD"
    finally:
        reader.rollback()
        reader.close()


def test_checkpoint_detects_corrupt_persisted_row(tmp_path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    store = CheckpointStore(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            INSERT INTO ingestion_checkpoints (
                pipeline_id, source, symbol, kind, timeframe, timeframe_key,
                last_event_time, last_sequence, last_batch_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "corrupt-pipeline",
                "mt5",
                "EURUSD",
                "tick",
                None,
                "",
                "not-a-timestamp",
                None,
                None,
                BASE.isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CheckpointCorruptionError, match="Invalid last_event_time"):
        store.get(CheckpointKey("corrupt-pipeline", "mt5", "EURUSD", DataKind.TICK))


def test_checkpoint_constructor_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        CheckpointStore(tmp_path / "bad.sqlite3", timeout_seconds=0)


def test_catalog_and_checkpoints_can_share_one_sqlite_database(tmp_path) -> None:
    from pathlib import Path

    from fxbot.data.catalog import DataCatalog, DatasetKey, DatasetRegistration

    database = tmp_path / "pipeline-state.sqlite3"
    catalog = DataCatalog(database)
    checkpoints = CheckpointStore(database)
    batch = catalog.register_batch(
        DatasetRegistration(
            key=DatasetKey(DataKind.TICK, "EURUSD"),
            start_time=BASE,
            end_time=BASE + timedelta(seconds=1),
            record_count=1,
            source="mt5",
            schema_version=1,
            content_hash="shared-db",
            files=(Path("part.parquet"),),
        )
    )

    saved = checkpoints.advance(
        tick_key(),
        last_event_time=BASE,
        last_batch_id=batch.batch_id,
    )

    assert catalog.get_batch(batch.batch_id) == batch
    assert saved.last_batch_id == batch.batch_id
