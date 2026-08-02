from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.clock import HistoricalClock, HistoricalClockError
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar, Tick

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_tick(seconds: int) -> Tick:
    return Tick(
        symbol="EURUSD",
        event_time=BASE + timedelta(seconds=seconds),
        bid=1.1 + seconds * 0.0001,
        ask=1.1002 + seconds * 0.0001,
    )


def make_bar() -> Bar:
    return Bar(
        symbol="EURUSD",
        open_time=BASE,
        timeframe=Timeframe.M1,
        bid=OHLC(1.1, 1.11, 1.09, 1.105),
        ask=OHLC(1.1002, 1.1102, 1.0902, 1.1052),
    )


def test_clock_sorts_records_stably() -> None:
    second = make_tick(2)
    first_a = make_tick(1)
    first_b = Tick(
        symbol="GBPUSD",
        event_time=first_a.event_time,
        bid=1.2,
        ask=1.2002,
    )
    events = HistoricalClock((second, first_a, first_b)).replay()
    assert [item.symbol for item in events] == ["EURUSD", "GBPUSD", "EURUSD"]
    assert [item.sequence for item in events] == [0, 1, 2]


def test_strict_clock_rejects_unsorted_input() -> None:
    with pytest.raises(HistoricalClockError, match="chronologically"):
        HistoricalClock((make_tick(2), make_tick(1)), strict_input_order=True)


def test_completed_bar_becomes_visible_at_close() -> None:
    bar = make_bar()
    event = HistoricalClock((bar,)).replay()[0]
    assert event.timestamp == BASE + timedelta(minutes=1)


def test_clock_filters_half_open_time_window() -> None:
    clock = HistoricalClock(
        (make_tick(0), make_tick(1), make_tick(2)),
        start=BASE + timedelta(seconds=1),
        end=BASE + timedelta(seconds=2),
    )
    assert len(clock) == 1
    assert clock.records[0].event_time == BASE + timedelta(seconds=1)


def test_clock_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="start must be earlier"):
        HistoricalClock((make_tick(1),), start=BASE, end=BASE)
