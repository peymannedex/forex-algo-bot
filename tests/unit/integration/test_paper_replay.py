import csv
from datetime import UTC, datetime

from fxbot.domain.enums import Timeframe
from fxbot.integration.replay import iter_paper_frames, load_replay_bars


def write_csv(path) -> None:
    rows = [
        {
            "timestamp": datetime(2026, 1, 5, tzinfo=UTC).isoformat(),
            "symbol": "EURUSD",
            "timeframe": "M15",
            "bid_open": "1.1000",
            "bid_high": "1.1010",
            "bid_low": "1.0990",
            "bid_close": "1.1005",
            "ask_open": "1.1002",
            "ask_high": "1.1012",
            "ask_low": "1.0992",
            "ask_close": "1.1007",
            "tick_volume": "100",
        },
        {
            "timestamp": datetime(2026, 1, 5, tzinfo=UTC).isoformat(),
            "symbol": "EURUSD",
            "timeframe": "M5",
            "bid_open": "1.1000",
            "bid_high": "1.1008",
            "bid_low": "1.0995",
            "bid_close": "1.1004",
            "ask_open": "1.1002",
            "ask_high": "1.1010",
            "ask_low": "1.0997",
            "ask_close": "1.1006",
            "tick_volume": "100",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_build_primary_frames(tmp_path) -> None:
    path = tmp_path / "replay.csv"
    write_csv(path)

    bars = load_replay_bars(path)
    frames = iter_paper_frames(bars, primary_timeframe=Timeframe.M5)

    assert len(bars) == 2
    assert len(frames) == 1
    assert frames[0].quote.symbol == "EURUSD"
    assert frames[0].context.primary_timeframe is Timeframe.M5


def test_missing_columns_rejected(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,symbol\n2026-01-01T00:00:00+00:00,EURUSD\n")

    try:
        load_replay_bars(path)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rows_are_sorted_by_close_time(tmp_path) -> None:
    path = tmp_path / "replay.csv"
    write_csv(path)

    bars = load_replay_bars(path)

    assert tuple(bar.close_time for bar in bars) == tuple(
        sorted(bar.close_time for bar in bars)
    )
