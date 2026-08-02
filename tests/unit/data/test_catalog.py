from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fxbot.data.catalog import (
    CatalogConflictError,
    CatalogCorruptionError,
    CoverageInterval,
    DataCatalog,
    DatasetKey,
    DatasetRegistration,
)
from fxbot.domain.enums import DataKind, Timeframe

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def registration(
    *,
    key: DatasetKey | None = None,
    start: datetime = BASE,
    end: datetime = BASE + timedelta(hours=1),
    content_hash: str = "hash-a",
    files: tuple[Path, ...] = (Path("part-a.parquet"),),
    source: str = "mt5",
) -> DatasetRegistration:
    return DatasetRegistration(
        key=key or DatasetKey(DataKind.TICK, "eurusd"),
        start_time=start,
        end_time=end,
        record_count=100,
        source=source,
        schema_version=1,
        content_hash=content_hash,
        files=files,
        created_at=BASE + timedelta(days=1),
    )


def test_catalog_initializes_reopens_and_preserves_batch(tmp_path) -> None:
    database = tmp_path / "catalog" / "catalog.sqlite3"
    catalog = DataCatalog(database)
    stored = catalog.register_batch(registration(), batch_id="batch-1")

    reopened = DataCatalog(database)

    assert stored.batch_id == "batch-1"
    assert stored.key == DatasetKey(DataKind.TICK, "EURUSD")
    assert stored.files == (Path("part-a.parquet"),)
    assert reopened.get_batch("batch-1") == stored
    assert reopened.contains_hash("HASH-A")
    assert reopened.get_by_hash("hash-a") == stored


def test_catalog_duplicate_hash_is_idempotent_but_conflicts_are_rejected(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.sqlite3")
    first = catalog.register_batch(registration(), batch_id="first")

    duplicate = catalog.register_batch(registration(), batch_id="ignored")

    assert duplicate == first
    assert len(catalog.list_batches()) == 1

    conflicting = registration(end=BASE + timedelta(hours=2))
    with pytest.raises(CatalogConflictError, match="different metadata"):
        catalog.register_batch(conflicting)


def test_dataset_keys_enforce_tick_and_bar_timeframes() -> None:
    assert DatasetKey(DataKind.TICK, "eurusd", Timeframe.TICK).timeframe is None
    assert DatasetKey(DataKind.BAR, "eurusd", Timeframe.M5).timeframe is Timeframe.M5

    with pytest.raises(ValueError, match="Tick datasets"):
        DatasetKey(DataKind.TICK, "EURUSD", Timeframe.M1)
    with pytest.raises(ValueError, match="Bar datasets"):
        DatasetKey(DataKind.BAR, "EURUSD")
    with pytest.raises(ValueError, match="non-tick"):
        DatasetKey(DataKind.BAR, "EURUSD", Timeframe.TICK)


def test_registration_rejects_naive_or_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        registration(start=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="earlier"):
        registration(start=BASE, end=BASE)
    with pytest.raises(ValueError, match="record_count"):
        DatasetRegistration(
            key=DatasetKey(DataKind.TICK, "EURUSD"),
            start_time=BASE,
            end_time=BASE + timedelta(seconds=1),
            record_count=0,
            source="mt5",
            schema_version=1,
            content_hash="hash",
            files=(Path("part.parquet"),),
        )
    with pytest.raises(ValueError, match="files"):
        DatasetRegistration(
            key=DatasetKey(DataKind.TICK, "EURUSD"),
            start_time=BASE,
            end_time=BASE + timedelta(seconds=1),
            record_count=1,
            source="mt5",
            schema_version=1,
            content_hash="hash",
            files=(),
        )


def test_catalog_lists_only_overlapping_half_open_batches(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.sqlite3")
    key = DatasetKey(DataKind.TICK, "EURUSD")
    first = catalog.register_batch(
        registration(key=key, end=BASE + timedelta(hours=1), content_hash="first")
    )
    catalog.register_batch(
        registration(
            key=key,
            start=BASE + timedelta(hours=1),
            end=BASE + timedelta(hours=2),
            content_hash="second",
        )
    )

    ending_at_boundary = catalog.list_batches(key=key, end=BASE + timedelta(hours=1))
    starting_at_boundary = catalog.list_batches(key=key, start=BASE + timedelta(hours=1))

    assert ending_at_boundary == (first,)
    assert [item.content_hash for item in starting_at_boundary] == ["second"]
    assert catalog.list_batches(source="other") == ()


def test_catalog_merges_adjacent_coverage_and_clips_query_window(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.sqlite3")
    key = DatasetKey(DataKind.BAR, "EURUSD", Timeframe.M1)
    for index, (start_hour, end_hour) in enumerate(((0, 1), (1, 2), (3, 4))):
        catalog.register_batch(
            registration(
                key=key,
                start=BASE + timedelta(hours=start_hour),
                end=BASE + timedelta(hours=end_hour),
                content_hash=f"bar-{index}",
            )
        )

    coverage = catalog.get_coverage(
        key,
        start=BASE + timedelta(minutes=30),
        end=BASE + timedelta(hours=3, minutes=30),
    )

    assert coverage == (
        CoverageInterval(BASE + timedelta(minutes=30), BASE + timedelta(hours=2)),
        CoverageInterval(BASE + timedelta(hours=3), BASE + timedelta(hours=3, minutes=30)),
    )
    assert coverage[0].contains(BASE + timedelta(hours=1))
    assert not coverage[0].contains(BASE + timedelta(hours=2))


def test_catalog_remove_does_not_delete_files(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.sqlite3")
    data_file = tmp_path / "part.parquet"
    data_file.write_bytes(b"parquet-placeholder")
    stored = catalog.register_batch(
        registration(files=(data_file,), content_hash="remove-me")
    )

    assert catalog.remove_batch(stored.batch_id)
    assert not catalog.remove_batch(stored.batch_id)
    assert catalog.get_batch(stored.batch_id) is None
    assert data_file.exists()


def test_catalog_wal_allows_write_while_reader_transaction_is_open(tmp_path) -> None:
    database = tmp_path / "catalog.sqlite3"
    catalog = DataCatalog(database)
    catalog.register_batch(registration(content_hash="existing"))

    reader = sqlite3.connect(database, isolation_level=None)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM dataset_batches").fetchone() == (1,)
        written = catalog.register_batch(
            registration(
                start=BASE + timedelta(hours=1),
                end=BASE + timedelta(hours=2),
                content_hash="new",
            )
        )
        assert written.content_hash == "new"
    finally:
        reader.rollback()
        reader.close()


def test_catalog_detects_corrupt_persisted_rows(tmp_path) -> None:
    database = tmp_path / "catalog.sqlite3"
    catalog = DataCatalog(database)
    connection = sqlite3.connect(database)
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
                "corrupt",
                "tick",
                "EURUSD",
                None,
                BASE.isoformat(),
                (BASE + timedelta(hours=1)).isoformat(),
                1,
                "test",
                1,
                "corrupt-hash",
                "{}",
                BASE.isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CatalogCorruptionError, match="files_json"):
        catalog.get_batch("corrupt")


def test_catalog_validates_query_window_and_constructor(tmp_path) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        DataCatalog(tmp_path / "bad.sqlite3", timeout_seconds=0)

    catalog = DataCatalog(tmp_path / "catalog.sqlite3")
    key = DatasetKey(DataKind.TICK, "EURUSD")
    with pytest.raises(ValueError, match="earlier"):
        catalog.list_batches(start=BASE, end=BASE)
    with pytest.raises(ValueError, match="earlier"):
        catalog.get_coverage(key, start=BASE, end=BASE)
