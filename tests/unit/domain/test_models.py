from datetime import UTC, datetime

import pytest

from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import OHLC, Bar, HistoricalDataRequest, SymbolSpec, Tick


def test_tick_normalizes_time_and_calculates_spread() -> None:
    tick = Tick(
        symbol="eurusd",
        event_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        bid=1.10000,
        ask=1.10012,
    )
    spec = SymbolSpec("EURUSD", "EUR", "USD", 5, 0.00001, 0.0001)

    assert tick.symbol == "EURUSD"
    assert tick.mid == pytest.approx(1.10006)
    assert tick.spread_pips(spec) == pytest.approx(1.2)


def test_tick_rejects_crossed_quote() -> None:
    with pytest.raises(ValueError, match="ask cannot be below bid"):
        Tick(
            symbol="EURUSD",
            event_time=datetime.now(UTC),
            bid=1.2,
            ask=1.1,
        )


def test_bar_preserves_bid_and_ask_ohlc() -> None:
    bar = Bar(
        symbol="EURUSD",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        timeframe=Timeframe.M1,
        bid=OHLC(1.1000, 1.1010, 1.0990, 1.1005),
        ask=OHLC(1.1002, 1.1012, 1.0992, 1.1007),
        tick_volume=42,
    )

    assert bar.spread_open == pytest.approx(0.0002)
    assert bar.mid.close == pytest.approx(1.1006)
    assert bar.close_time == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


def test_historical_bar_request_requires_timeframe() -> None:
    with pytest.raises(ValueError, match="require a non-tick timeframe"):
        HistoricalDataRequest(symbol="EURUSD", kind=DataKind.BAR)
