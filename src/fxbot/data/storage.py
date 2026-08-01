"""Atomic Parquet partition storage for canonical market-data records.

The store writes immutable Parquet part files under Hive-style directories:

``kind=tick/symbol=EURUSD/year=2026/month=07/part-....parquet``
``kind=bar/symbol=EURUSD/timeframe=1m/year=2026/month=07/part-....parquet``

Files are written to a temporary path in the destination directory and moved
atomically into place.  Existing part files are never modified during append,
which keeps readers safe and makes failed writes recoverable.
"""

from __future__ import annotations

import heapq
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import OHLC, Bar, HistoricalDataRequest, MarketDataRecord, Tick

_SCHEMA_VERSION = 1
_SAFE_PARTITION_VALUE = re.compile(r"^[A-Z0-9._-]+$")
_RecordT = TypeVar("_RecordT", Tick, Bar)


class ParquetStorageError(RuntimeError):
    """Raised when market data cannot be persisted or reconstructed safely."""


@dataclass(frozen=True, slots=True)
class PartitionRef:
    """Logical partition containing one calendar month of one data stream."""

    kind: DataKind
    symbol: str
    year: int
    month: int
    timeframe: Timeframe | None = None

    def __post_init__(self) -> None:
        normalized = self.symbol.strip().upper()
        if not _SAFE_PARTITION_VALUE.fullmatch(normalized):
            raise ValueError(f"Unsafe partition symbol: {self.symbol!r}")
        if not 1 <= self.month <= 12:
            raise ValueError("month must be in the range 1..12")
        if self.year < 1970:
            raise ValueError("year must be 1970 or later")
        object.__setattr__(self, "symbol", normalized)
        object.__setattr__(self, "kind", DataKind(self.kind))
        if self.kind is DataKind.TICK:
            if self.timeframe not in (None, Timeframe.TICK):
                raise ValueError("Tick partitions cannot have a bar timeframe")
            object.__setattr__(self, "timeframe", None)
        else:
            if self.timeframe in (None, Timeframe.TICK):
                raise ValueError("Bar partitions require a non-tick timeframe")
            object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe))


@dataclass(frozen=True, slots=True)
class StorageWriteResult:
    """Summary of an append operation."""

    records_written: int
    files_created: tuple[Path, ...]
    partitions: tuple[PartitionRef, ...]


class ParquetPartitionStore:
    """Append and query canonical ticks and bars in monthly Parquet partitions.

    Args:
        root: Root directory for the partition tree.
        compression: PyArrow compression codec. ``zstd`` gives a strong balance
            of compression ratio and read speed for market data.
        row_group_size: Maximum rows per Parquet row group.
        read_batch_size: Rows materialized at a time when reading a part file.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        compression: str = "zstd",
        row_group_size: int = 250_000,
        read_batch_size: int = 65_536,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.compression = compression
        self.row_group_size = row_group_size
        self.read_batch_size = read_batch_size
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        if read_batch_size <= 0:
            raise ValueError("read_batch_size must be positive")

    def append(self, records: Iterable[MarketDataRecord]) -> StorageWriteResult:
        """Append ticks and/or bars, creating one atomic file per partition."""

        grouped: dict[PartitionRef, list[MarketDataRecord]] = defaultdict(list)
        count = 0
        for record in records:
            partition = self._partition_for(record)
            grouped[partition].append(record)
            count += 1

        created: list[Path] = []
        for partition in sorted(grouped, key=self._partition_sort_key):
            partition_records = grouped[partition]
            partition_records.sort(key=self._record_sort_key)
            created.append(self._write_partition_file(partition, partition_records))

        return StorageWriteResult(
            records_written=count,
            files_created=tuple(created),
            partitions=tuple(sorted(grouped, key=self._partition_sort_key)),
        )

    def append_ticks(self, ticks: Iterable[Tick]) -> StorageWriteResult:
        """Append tick records and reject accidental bar input."""

        validated = self._validate_type(ticks, Tick, "append_ticks")
        return self.append(validated)

    def append_bars(self, bars: Iterable[Bar]) -> StorageWriteResult:
        """Append bar records and reject accidental tick input."""

        validated = self._validate_type(bars, Bar, "append_bars")
        return self.append(validated)

    def iter_ticks(self, request: HistoricalDataRequest) -> Iterator[Tick]:
        """Yield ticks in chronological order using partition and row filtering."""

        if request.kind is not DataKind.TICK:
            raise ValueError("iter_ticks requires a tick HistoricalDataRequest")
        files = self._candidate_files(request)
        streams = [self._iter_tick_file(path, request) for path in files]
        yield from heapq.merge(
            *streams,
            key=lambda item: (item.event_time, item.sequence if item.sequence is not None else -1),
        )

    def iter_bars(self, request: HistoricalDataRequest) -> Iterator[Bar]:
        """Yield bars in chronological order using partition and row filtering."""

        if request.kind is not DataKind.BAR:
            raise ValueError("iter_bars requires a bar HistoricalDataRequest")
        files = self._candidate_files(request)
        streams = [self._iter_bar_file(path, request) for path in files]
        yield from heapq.merge(*streams, key=lambda item: item.open_time)

    def list_partitions(
        self,
        *,
        kind: DataKind | None = None,
        symbol: str | None = None,
        timeframe: Timeframe | str | None = None,
    ) -> tuple[PartitionRef, ...]:
        """Inspect partitions that contain at least one Parquet part file."""

        requested_kind = DataKind(kind) if kind is not None else None
        normalized_symbol = symbol.strip().upper() if symbol is not None else None
        parsed_timeframe = Timeframe.parse(timeframe) if timeframe is not None else None
        found: set[PartitionRef] = set()

        for path in self.root.rglob("*.parquet") if self.root.exists() else ():
            partition = self._partition_from_path(path)
            if partition is None:
                continue
            if requested_kind is not None and partition.kind is not requested_kind:
                continue
            if normalized_symbol is not None and partition.symbol != normalized_symbol:
                continue
            if parsed_timeframe is not None and partition.timeframe is not parsed_timeframe:
                continue
            found.add(partition)
        return tuple(sorted(found, key=self._partition_sort_key))

    def _write_partition_file(
        self,
        partition: PartitionRef,
        records: Sequence[MarketDataRecord],
    ) -> Path:
        pa, pq = _import_pyarrow()
        directory = self._partition_path(partition)
        directory.mkdir(parents=True, exist_ok=True)

        rows = (
            [self._tick_to_row(record) for record in records if isinstance(record, Tick)]
            if partition.kind is DataKind.TICK
            else [self._bar_to_row(record) for record in records if isinstance(record, Bar)]
        )
        schema = _tick_schema(pa) if partition.kind is DataKind.TICK else _bar_schema(pa)
        table = pa.Table.from_pylist(rows, schema=schema)
        metadata = dict(table.schema.metadata or {})
        metadata[b"fxbot_schema_version"] = str(_SCHEMA_VERSION).encode("ascii")
        table = table.replace_schema_metadata(metadata)

        first_ns = int(self._record_time(records[0]).timestamp() * 1_000_000_000)
        last_ns = int(self._record_time(records[-1]).timestamp() * 1_000_000_000)
        filename = f"part-{first_ns}-{last_ns}-{uuid4().hex}.parquet"
        destination = directory / filename
        temporary = directory / f".{filename}.tmp"

        try:
            pq.write_table(
                table,
                temporary,
                compression=self.compression,
                row_group_size=self.row_group_size,
                use_dictionary=(
                    ["symbol", "source"]
                    if partition.kind is DataKind.TICK
                    else ["symbol", "source", "timeframe"]
                ),
                write_statistics=True,
            )
            os.replace(temporary, destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise ParquetStorageError(
                f"Failed writing partition {partition} to {destination}: {exc}"
            ) from exc
        return destination

    def _candidate_files(self, request: HistoricalDataRequest) -> tuple[Path, ...]:
        symbol = self._safe_symbol(request.symbol)
        base = self.root / f"kind={request.kind.value}" / f"symbol={symbol}"
        if request.kind is DataKind.BAR:
            assert request.timeframe is not None
            base /= f"timeframe={request.timeframe.value}"
        if not base.exists():
            return ()

        if request.start is None or request.end is None:
            return tuple(sorted(base.rglob("*.parquet")))

        paths: list[Path] = []
        for year, month in _iter_months(request.start, request.end):
            month_dir = base / f"year={year:04d}" / f"month={month:02d}"
            paths.extend(sorted(month_dir.glob("*.parquet")))
        return tuple(paths)

    def _iter_tick_file(
        self,
        path: Path,
        request: HistoricalDataRequest,
    ) -> Iterator[Tick]:
        for row in self._iter_file_rows(path):
            try:
                tick = self._row_to_tick(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ParquetStorageError(f"Invalid tick row in {path}: {exc}") from exc
            if tick.symbol == request.symbol and request.contains(tick.event_time):
                yield tick

    def _iter_bar_file(
        self,
        path: Path,
        request: HistoricalDataRequest,
    ) -> Iterator[Bar]:
        for row in self._iter_file_rows(path):
            try:
                bar = self._row_to_bar(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ParquetStorageError(f"Invalid bar row in {path}: {exc}") from exc
            if (
                bar.symbol == request.symbol
                and bar.timeframe is request.timeframe
                and request.contains(bar.open_time)
            ):
                yield bar

    def _iter_file_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        _, pq = _import_pyarrow()
        try:
            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.schema_arrow.metadata or {}
            version = int(metadata.get(b"fxbot_schema_version", b"0"))
            if version != _SCHEMA_VERSION:
                raise ParquetStorageError(
                    f"Unsupported schema version {version} in {path}; expected {_SCHEMA_VERSION}"
                )
            for batch in parquet_file.iter_batches(batch_size=self.read_batch_size):
                yield from batch.to_pylist()
        except ParquetStorageError:
            raise
        except Exception as exc:
            raise ParquetStorageError(f"Failed reading {path}: {exc}") from exc

    def _partition_path(self, partition: PartitionRef) -> Path:
        path = (
            self.root
            / f"kind={partition.kind.value}"
            / f"symbol={self._safe_symbol(partition.symbol)}"
        )
        if partition.kind is DataKind.BAR:
            assert partition.timeframe is not None
            path /= f"timeframe={partition.timeframe.value}"
        return path / f"year={partition.year:04d}" / f"month={partition.month:02d}"

    @staticmethod
    def _partition_for(record: MarketDataRecord) -> PartitionRef:
        timestamp = ParquetPartitionStore._record_time(record)
        symbol = ParquetPartitionStore._safe_symbol(record.symbol)
        return PartitionRef(
            kind=DataKind.TICK if isinstance(record, Tick) else DataKind.BAR,
            symbol=symbol,
            year=timestamp.year,
            month=timestamp.month,
            timeframe=None if isinstance(record, Tick) else record.timeframe,
        )

    @staticmethod
    def _partition_from_path(path: Path) -> PartitionRef | None:
        values: dict[str, str] = {}
        for part in path.parts:
            if "=" in part:
                key, value = part.split("=", 1)
                values[key] = value
        try:
            kind = DataKind(values["kind"])
            return PartitionRef(
                kind=kind,
                symbol=values["symbol"],
                timeframe=values.get("timeframe"),  # type: ignore[arg-type]
                year=int(values["year"]),
                month=int(values["month"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _record_time(record: MarketDataRecord) -> datetime:
        return record.event_time if isinstance(record, Tick) else record.open_time

    @staticmethod
    def _record_sort_key(record: MarketDataRecord) -> tuple[datetime, int]:
        sequence = (
            record.sequence
            if isinstance(record, Tick) and record.sequence is not None
            else -1
        )
        return ParquetPartitionStore._record_time(record), sequence

    @staticmethod
    def _partition_sort_key(partition: PartitionRef) -> tuple[str, str, str, int, int]:
        return (
            partition.kind.value,
            partition.symbol,
            partition.timeframe.value if partition.timeframe is not None else "",
            partition.year,
            partition.month,
        )

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not _SAFE_PARTITION_VALUE.fullmatch(normalized):
            raise ParquetStorageError(f"Unsafe partition symbol: {symbol!r}")
        return normalized

    @staticmethod
    def _validate_type(
        records: Iterable[_RecordT],
        expected: type[_RecordT],
        operation: str,
    ) -> Iterator[_RecordT]:
        for record in records:
            if not isinstance(record, expected):
                raise TypeError(f"{operation} received {type(record).__name__}")
            yield record

    @staticmethod
    def _tick_to_row(tick: Tick) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "symbol": tick.symbol,
            "event_time": tick.event_time,
            "bid": tick.bid,
            "ask": tick.ask,
            "bid_size": tick.bid_size,
            "ask_size": tick.ask_size,
            "source": tick.source,
            "sequence": tick.sequence,
            "received_time": tick.received_time,
        }

    @staticmethod
    def _bar_to_row(bar: Bar) -> dict[str, Any]:
        mid = bar.mid
        return {
            "schema_version": _SCHEMA_VERSION,
            "symbol": bar.symbol,
            "open_time": bar.open_time,
            "timeframe": bar.timeframe.value,
            "bid_open": bar.bid.open,
            "bid_high": bar.bid.high,
            "bid_low": bar.bid.low,
            "bid_close": bar.bid.close,
            "ask_open": bar.ask.open,
            "ask_high": bar.ask.high,
            "ask_low": bar.ask.low,
            "ask_close": bar.ask.close,
            "mid_open": mid.open,
            "mid_high": mid.high,
            "mid_low": mid.low,
            "mid_close": mid.close,
            "tick_volume": bar.tick_volume,
            "real_volume": bar.real_volume,
            "source": bar.source,
            "complete": bar.complete,
        }

    @staticmethod
    def _row_to_tick(row: dict[str, Any]) -> Tick:
        return Tick(
            symbol=str(row["symbol"]),
            event_time=_as_utc_datetime(row["event_time"]),
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            bid_size=_optional_float(row.get("bid_size")),
            ask_size=_optional_float(row.get("ask_size")),
            source=str(row["source"]),
            sequence=_optional_int(row.get("sequence")),
            received_time=_optional_datetime(row.get("received_time")),
        )

    @staticmethod
    def _row_to_bar(row: dict[str, Any]) -> Bar:
        return Bar(
            symbol=str(row["symbol"]),
            open_time=_as_utc_datetime(row["open_time"]),
            timeframe=Timeframe.parse(str(row["timeframe"])),
            bid=OHLC(
                float(row["bid_open"]),
                float(row["bid_high"]),
                float(row["bid_low"]),
                float(row["bid_close"]),
            ),
            ask=OHLC(
                float(row["ask_open"]),
                float(row["ask_high"]),
                float(row["ask_low"]),
                float(row["ask_close"]),
            ),
            mid_ohlc=OHLC(
                float(row["mid_open"]),
                float(row["mid_high"]),
                float(row["mid_low"]),
                float(row["mid_close"]),
            ),
            tick_volume=int(row["tick_volume"]),
            real_volume=_optional_float(row.get("real_volume")),
            source=str(row["source"]),
            complete=bool(row["complete"]),
        )


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ParquetStorageError(
            "pyarrow is required for ParquetPartitionStore; install runtime dependencies"
        ) from exc
    return pa, pq


def _tick_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("schema_version", pa.int16()),
            ("symbol", pa.string()),
            ("event_time", pa.timestamp("us", tz="UTC")),
            ("bid", pa.float64()),
            ("ask", pa.float64()),
            ("bid_size", pa.float64()),
            ("ask_size", pa.float64()),
            ("source", pa.string()),
            ("sequence", pa.int64()),
            ("received_time", pa.timestamp("us", tz="UTC")),
        ]
    )


def _bar_schema(pa: Any) -> Any:
    fields = [
        ("schema_version", pa.int16()),
        ("symbol", pa.string()),
        ("open_time", pa.timestamp("us", tz="UTC")),
        ("timeframe", pa.string()),
    ]
    fields.extend((name, pa.float64()) for name in (
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
        "mid_open", "mid_high", "mid_low", "mid_close",
    ))
    fields.extend(
        [
            ("tick_volume", pa.int64()),
            ("real_volume", pa.float64()),
            ("source", pa.string()),
            ("complete", pa.bool_()),
        ]
    )
    return pa.schema(fields)


def _iter_months(start: datetime, end: datetime) -> Iterator[tuple[int, int]]:
    """Yield all calendar months intersecting the half-open UTC interval."""

    current = start.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = end.astimezone(UTC)
    while current < last:
        yield current.year, current.month
        current = (
            current.replace(year=current.year + 1, month=1)
            if current.month == 12
            else current.replace(month=current.month + 1)
        )


def _as_utc_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime, received {type(value)!r}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Stored timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else _as_utc_datetime(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[call-overload]
