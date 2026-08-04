from datetime import UTC, datetime

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar, Tick
from fxbot.integration.soak import LivePaperFrameAssembler


def bar(
    timeframe: Timeframe,
    minute: int,
    *,
    complete: bool = True,
) -> Bar:
    return Bar(
        symbol="EURUSD",
        open_time=datetime(2026, 1, 5, 14, minute, tzinfo=UTC),
        timeframe=timeframe,
        bid=OHLC(1.1, 1.101, 1.099, 1.1005),
        ask=OHLC(1.1002, 1.1012, 1.0992, 1.1007),
        complete=complete,
    )


def test_assembler_emits_only_completed_primary_frames() -> None:
    assembler = LivePaperFrameAssembler(
        primary_timeframe=Timeframe.M5,
        required_timeframes=(Timeframe.M5, Timeframe.M15),
        history_limit=100,
    )

    assert assembler.ingest(bar(Timeframe.M15, 0)) is None
    assert assembler.ingest(bar(Timeframe.M5, 0, complete=False)) is None

    frame = assembler.ingest(bar(Timeframe.M5, 0))

    assert frame is not None
    assert frame.quote.timestamp == datetime(
        2026,
        1,
        5,
        14,
        5,
        tzinfo=UTC,
    )


def test_assembler_suppresses_duplicate_bars_and_ticks() -> None:
    assembler = LivePaperFrameAssembler(
        primary_timeframe=Timeframe.M5,
        required_timeframes=(Timeframe.M5, Timeframe.M15),
        history_limit=100,
    )
    item = bar(Timeframe.M5, 0)

    assert assembler.ingest(item) is not None
    assert assembler.ingest(item) is None
    assert (
        assembler.ingest(
            Tick(
                "EURUSD",
                datetime(2026, 1, 5, 14, 1, tzinfo=UTC),
                1.1,
                1.1002,
            )
        )
        is None
    )
