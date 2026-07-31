from datetime import UTC, datetime

import pytest

from fxbot.data.storage import ParquetPartitionStore, ParquetStorageError
from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import Bar, HistoricalDataRequest, OHLC, Tick


def test_store_partitions_ticks_by_symbol_year_and_month_and_reads_sorted(tmp_path) -> None:
    store = ParquetPartitionStore(tmp_path / "market-data", row_group_size=2)
    february = Tick(
        "EURUSD",
        datetime(2026, 2, 1, tzinfo=UTC),
        1.2000,
        1.2002,
        sequence=2,
        source="test",
    )
    january = Tick(
        "EURUSD",
        datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        1.1000,
        1.1002,
        sequence=1,
        source="test",
    )

    result = store.append_ticks([february, january])

    assert result.records_written == 2
    assert len(result.files_created) == 2
    assert {partition.month for partition in result.partitions} == {1, 2}
    assert all(path.exists() for path in result.files_created)
    assert not list((tmp_path / "market-data").rglob("*.tmp"))

    request = HistoricalDataRequest(
        symbol="EURUSD",
        kind=DataKind.TICK,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 3, 1, tzinfo=UTC),
    )
    restored = list(store.iter_ticks(request))

    assert restored == [january, february]


def test_store_round_trips_explicit_mid_ohlc_for_resampled_bars(tmp_path) -> None:
    store = ParquetPartitionStore(tmp_path / "market-data")
    bar = Bar(
        symbol="EURUSD",
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        timeframe=Timeframe.M1,
        bid=OHLC(1.0, 1.3, 0.9, 1.2),
        ask=OHLC(1.4, 1.4, 1.1, 1.21),
        mid_ohlc=OHLC(1.2, 1.305, 1.0, 1.205),
        tick_volume=12,
        source="resampled:test",
        complete=True,
    )

    store.append_bars([bar])
    request = HistoricalDataRequest(
        symbol="EURUSD",
        kind=DataKind.BAR,
        timeframe=Timeframe.M1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )

    restored = list(store.iter_bars(request))

    assert restored == [bar]
    assert restored[0].mid.high == pytest.approx(1.305)
    partitions = store.list_partitions(kind=DataKind.BAR, symbol="EURUSD")
    assert len(partitions) == 1
    assert partitions[0].timeframe is Timeframe.M1


def test_store_honors_half_open_query_end(tmp_path) -> None:
    store = ParquetPartitionStore(tmp_path / "market-data")
    at_end = Tick("EURUSD", datetime(2026, 2, 1, tzinfo=UTC), 1.2, 1.2001)
    before_end = Tick(
        "EURUSD", datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC), 1.1, 1.1001
    )
    store.append_ticks([at_end, before_end])

    request = HistoricalDataRequest(
        symbol="EURUSD",
        kind=DataKind.TICK,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 2, 1, tzinfo=UTC),
    )

    assert list(store.iter_ticks(request)) == [before_end]


def test_store_rejects_unsafe_partition_symbol(tmp_path) -> None:
    store = ParquetPartitionStore(tmp_path / "market-data")
    tick = Tick("EUR/USD", datetime(2026, 1, 1, tzinfo=UTC), 1.1, 1.1001)

    with pytest.raises(ParquetStorageError, match="Unsafe partition symbol"):
        store.append_ticks([tick])
