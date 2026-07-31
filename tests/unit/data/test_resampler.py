from datetime import UTC, datetime, timedelta

import pytest

from fxbot.data.resampler import (
    LateTickPolicy,
    OutOfOrderTickError,
    TickBarResampler,
    align_open_time,
)
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import Tick


def test_resampler_aggregates_aligned_bid_ask_and_true_mid() -> None:
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    ticks = [
        Tick("EURUSD", start + timedelta(seconds=2), 1.0000, 1.4000, source="feed"),
        Tick("EURUSD", start + timedelta(seconds=20), 1.3000, 1.3100, source="feed"),
        Tick("EURUSD", start + timedelta(minutes=1), 1.2000, 1.2200, source="feed"),
    ]
    resampler = TickBarResampler(Timeframe.M1)

    completed = list(resampler.resample(ticks))

    assert len(completed) == 2
    first = completed[0]
    assert first.complete is True
    assert first.open_time == start
    assert first.tick_volume == 2
    assert first.bid.open == pytest.approx(1.0000)
    assert first.bid.high == pytest.approx(1.3000)
    assert first.ask.open == pytest.approx(1.4000)
    assert first.ask.low == pytest.approx(1.3100)
    # True tick-mid high is 1.305. Averaging independent bid/ask highs
    # would incorrectly produce 1.35, a price that never existed.
    assert first.mid.high == pytest.approx(1.3050)
    assert first.mid.high != pytest.approx((first.bid.high + first.ask.high) / 2)
    assert completed[1].complete is False


def test_weekly_alignment_uses_monday_utc() -> None:
    sunday = datetime(2026, 1, 11, 23, 59, tzinfo=UTC)

    assert align_open_time(sunday, Timeframe.W1) == datetime(
        2026, 1, 5, tzinfo=UTC
    )


def test_resampler_rejects_out_of_order_ticks_by_default() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    resampler = TickBarResampler(Timeframe.M1)
    resampler.update(Tick("EURUSD", start + timedelta(seconds=2), 1.1, 1.1001))

    with pytest.raises(OutOfOrderTickError):
        resampler.update(Tick("EURUSD", start + timedelta(seconds=1), 1.1, 1.1001))


def test_resampler_can_explicitly_drop_late_ticks() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    resampler = TickBarResampler(
        Timeframe.M1,
        late_tick_policy=LateTickPolicy.DROP,
    )
    resampler.update(Tick("EURUSD", start + timedelta(seconds=2), 1.1, 1.1001))

    emitted = resampler.update(
        Tick("EURUSD", start + timedelta(seconds=1), 1.1, 1.1001)
    )

    assert emitted == ()
    assert resampler.late_ticks_dropped == 1
