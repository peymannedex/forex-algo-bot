"""Enumerations used across the platform."""

from __future__ import annotations

from enum import StrEnum


class DataKind(StrEnum):
    """Supported market-data record families."""

    TICK = "tick"
    BAR = "bar"


class Timeframe(StrEnum):
    """Canonical timeframe identifiers.

    Timeframes are intentionally broker-neutral. Broker adapters are responsible
    for translating these values into platform-specific constants.
    """

    TICK = "tick"
    S1 = "1s"
    S5 = "5s"
    S10 = "10s"
    S30 = "30s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int | None:
        """Return the fixed interval in seconds, or ``None`` for tick data."""

        values: dict[Timeframe, int | None] = {
            Timeframe.TICK: None,
            Timeframe.S1: 1,
            Timeframe.S5: 5,
            Timeframe.S10: 10,
            Timeframe.S30: 30,
            Timeframe.M1: 60,
            Timeframe.M5: 5 * 60,
            Timeframe.M15: 15 * 60,
            Timeframe.M30: 30 * 60,
            Timeframe.H1: 60 * 60,
            Timeframe.H4: 4 * 60 * 60,
            Timeframe.D1: 24 * 60 * 60,
            Timeframe.W1: 7 * 24 * 60 * 60,
        }
        return values[self]

    @classmethod
    def parse(cls, value: str | Timeframe) -> Timeframe:
        """Parse common broker/data-vendor timeframe spellings."""

        if isinstance(value, Timeframe):
            return value

        normalized = value.strip().lower().replace("_", "")
        aliases = {
            "tick": cls.TICK,
            "t": cls.TICK,
            "s1": cls.S1,
            "1s": cls.S1,
            "s5": cls.S5,
            "5s": cls.S5,
            "s10": cls.S10,
            "10s": cls.S10,
            "s30": cls.S30,
            "30s": cls.S30,
            "m1": cls.M1,
            "1m": cls.M1,
            "m5": cls.M5,
            "5m": cls.M5,
            "m15": cls.M15,
            "15m": cls.M15,
            "m30": cls.M30,
            "30m": cls.M30,
            "h1": cls.H1,
            "1h": cls.H1,
            "h4": cls.H4,
            "4h": cls.H4,
            "d1": cls.D1,
            "1d": cls.D1,
            "w1": cls.W1,
            "1w": cls.W1,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported timeframe: {value!r}") from exc


class ParseErrorPolicy(StrEnum):
    """How an adapter handles malformed source rows."""

    RAISE = "raise"
    SKIP = "skip"


class QueueOverflowPolicy(StrEnum):
    """Behavior when a bounded live-data queue is full."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    RAISE = "raise"
