"""Deterministic historical event clock with look-ahead-safe timestamps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

from fxbot.backtest.events import MarketDataRecord, MarketEvent, market_record_time


class HistoricalClockError(ValueError):
    """Raised when a historical feed violates configured ordering rules."""


class HistoricalClock:
    """Replay ticks and bars in stable chronological order.

    Completed bars become visible at their close time. Incomplete bars become
    visible at their open time. Equal timestamps retain source order.
    """

    def __init__(
        self,
        records: Iterable[MarketDataRecord],
        *,
        strict_input_order: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        indexed = tuple(enumerate(records))
        self.start = self._optional_utc(start, "start")
        self.end = self._optional_utc(end, "end")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start must be earlier than end")

        if strict_input_order:
            previous: datetime | None = None
            for _, record in indexed:
                current = market_record_time(record)
                if previous is not None and current < previous:
                    raise HistoricalClockError("Historical records are not chronologically ordered")
                previous = current
            ordered = indexed
        else:
            ordered = tuple(sorted(indexed, key=lambda item: (market_record_time(item[1]), item[0])))

        self._records = tuple(
            record
            for _, record in ordered
            if (self.start is None or market_record_time(record) >= self.start)
            and (self.end is None or market_record_time(record) < self.end)
        )

    @staticmethod
    def _optional_utc(value: datetime | None, field_name: str) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def records(self) -> tuple[MarketDataRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MarketEvent]:
        for sequence, record in enumerate(self._records):
            yield MarketEvent(
                sequence=sequence,
                timestamp=market_record_time(record),
                record=record,
            )

    def replay(self) -> tuple[MarketEvent, ...]:
        return tuple(iter(self))
