"""Deterministic tick-to-bar resampling.

The resampler aggregates bid, ask, and true tick-level mid prices in the same
UTC-aligned bucket.  Computing the mid high/low directly from tick mids is
important: averaging the independently observed bid-high and ask-high can
produce a value that never existed in the market because those extrema may
have occurred on different ticks.

The implementation intentionally does not synthesize empty bars.  A missing
bucket therefore remains an observable data gap instead of being hidden by a
forward-filled candle.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import Bar, OHLC, Tick


class ResamplingError(ValueError):
    """Base exception for invalid tick streams or resampling configuration."""


class OutOfOrderTickError(ResamplingError):
    """Raised when a tick arrives earlier than the latest tick for its symbol."""


class LateTickPolicy(StrEnum):
    """Policy applied when a tick belongs to an already closed time bucket."""

    RAISE = "raise"
    DROP = "drop"


_WEEK_ANCHOR = datetime(1970, 1, 5, tzinfo=UTC)  # Monday, ISO-week alignment.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def align_open_time(timestamp: datetime, timeframe: Timeframe | str) -> datetime:
    """Floor ``timestamp`` to the canonical UTC opening time for ``timeframe``.

    Weekly bars are aligned to Monday 00:00 UTC.  All other supported
    timeframes are aligned to Unix-epoch multiples, which also gives the
    expected UTC boundaries for seconds, minutes, hours, and daily bars.
    """

    parsed = Timeframe.parse(timeframe)
    seconds = parsed.seconds
    if seconds is None:
        raise ResamplingError("Tick timeframe cannot be used for OHLC resampling")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ResamplingError("timestamp must be timezone-aware")

    value = timestamp.astimezone(UTC)
    interval = timedelta(seconds=seconds)
    anchor = _WEEK_ANCHOR if parsed is Timeframe.W1 else _EPOCH
    bucket_count = (value - anchor) // interval
    return anchor + bucket_count * interval


@dataclass(slots=True)
class _PriceAccumulator:
    """Mutable OHLC accumulator used only inside a single bucket."""

    open: float
    high: float
    low: float
    close: float

    @classmethod
    def start(cls, value: float) -> _PriceAccumulator:
        return cls(open=value, high=value, low=value, close=value)

    def update(self, value: float) -> None:
        self.high = max(self.high, value)
        self.low = min(self.low, value)
        self.close = value

    def freeze(self) -> OHLC:
        return OHLC(open=self.open, high=self.high, low=self.low, close=self.close)


@dataclass(slots=True)
class _BucketAccumulator:
    symbol: str
    open_time: datetime
    timeframe: Timeframe
    bid: _PriceAccumulator
    ask: _PriceAccumulator
    mid: _PriceAccumulator
    tick_volume: int
    sources: set[str]

    @classmethod
    def start(cls, tick: Tick, timeframe: Timeframe, open_time: datetime) -> _BucketAccumulator:
        return cls(
            symbol=tick.symbol,
            open_time=open_time,
            timeframe=timeframe,
            bid=_PriceAccumulator.start(tick.bid),
            ask=_PriceAccumulator.start(tick.ask),
            mid=_PriceAccumulator.start(tick.mid),
            tick_volume=1,
            sources={tick.source},
        )

    def update(self, tick: Tick) -> None:
        self.bid.update(tick.bid)
        self.ask.update(tick.ask)
        self.mid.update(tick.mid)
        self.tick_volume += 1
        self.sources.add(tick.source)

    def freeze(self, *, complete: bool) -> Bar:
        source_name = next(iter(self.sources)) if len(self.sources) == 1 else "mixed"
        return Bar(
            symbol=self.symbol,
            open_time=self.open_time,
            timeframe=self.timeframe,
            bid=self.bid.freeze(),
            ask=self.ask.freeze(),
            mid_ohlc=self.mid.freeze(),
            tick_volume=self.tick_volume,
            source=f"resampled:{source_name}",
            complete=complete,
        )


class TickBarResampler:
    """Incrementally convert two-sided ticks into UTC-aligned OHLC bars.

    One instance may process interleaved symbols.  Ordering is enforced per
    symbol, not globally, so a later EURUSD tick does not prevent an older
    GBPUSD tick from being accepted if GBPUSD itself remains ordered.

    Args:
        timeframe: Target non-tick timeframe.
        late_tick_policy: Raise on late ticks or explicitly drop them.
    """

    def __init__(
        self,
        timeframe: Timeframe | str,
        *,
        late_tick_policy: LateTickPolicy = LateTickPolicy.RAISE,
    ) -> None:
        self._timeframe = Timeframe.parse(timeframe)
        if self._timeframe is Timeframe.TICK:
            raise ResamplingError("Target timeframe must be an OHLC timeframe")
        self._late_tick_policy = LateTickPolicy(late_tick_policy)
        self._buckets: dict[str, _BucketAccumulator] = {}
        self._last_event_time: dict[str, datetime] = {}
        self._late_ticks_dropped = 0

    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def late_ticks_dropped(self) -> int:
        return self._late_ticks_dropped

    def update(self, tick: Tick) -> tuple[Bar, ...]:
        """Consume one tick and return any bar completed by that tick.

        At most one bar is emitted because empty buckets are not fabricated.
        The currently open bar remains internal until a later bucket arrives or
        :meth:`flush` is called.
        """

        previous_time = self._last_event_time.get(tick.symbol)
        if previous_time is not None and tick.event_time < previous_time:
            return self._handle_late_tick(tick, previous_time)

        bucket_open = align_open_time(tick.event_time, self._timeframe)
        current = self._buckets.get(tick.symbol)
        self._last_event_time[tick.symbol] = tick.event_time

        if current is None:
            self._buckets[tick.symbol] = _BucketAccumulator.start(
                tick, self._timeframe, bucket_open
            )
            return ()

        if bucket_open < current.open_time:
            return self._handle_late_tick(tick, previous_time or current.open_time)

        if bucket_open == current.open_time:
            current.update(tick)
            return ()

        completed = current.freeze(complete=True)
        self._buckets[tick.symbol] = _BucketAccumulator.start(
            tick, self._timeframe, bucket_open
        )
        return (completed,)

    def flush(self, *, complete: bool = False) -> tuple[Bar, ...]:
        """Emit and clear all open buckets.

        ``complete=False`` is the safe default because the caller may be
        stopping in the middle of a bar.  Historical importers that know the
        source interval is closed may explicitly set it to ``True``.
        """

        bars = tuple(
            bucket.freeze(complete=complete)
            for bucket in sorted(
                self._buckets.values(), key=lambda item: (item.open_time, item.symbol)
            )
        )
        self._buckets.clear()
        self._last_event_time.clear()
        return bars

    def resample(
        self,
        ticks: Iterable[Tick],
        *,
        final_bar_complete: bool = False,
    ) -> Iterator[Bar]:
        """Resample an iterable and then flush its final open buckets."""

        for tick in ticks:
            yield from self.update(tick)
        yield from self.flush(complete=final_bar_complete)

    def _handle_late_tick(self, tick: Tick, previous_time: datetime) -> tuple[Bar, ...]:
        if self._late_tick_policy is LateTickPolicy.DROP:
            self._late_ticks_dropped += 1
            return ()
        raise OutOfOrderTickError(
            f"Late tick for {tick.symbol}: {tick.event_time.isoformat()} is earlier "
            f"than {previous_time.isoformat()}"
        )
