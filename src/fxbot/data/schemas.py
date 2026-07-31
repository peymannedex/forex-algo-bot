"""Declarative mappings from vendor columns to canonical domain fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fxbot.domain.enums import Timeframe


class BarQuoteMode(StrEnum):
    """How a source represents two-sided OHLC prices."""

    BID_ASK = "bid_ask"
    BID_PLUS_SPREAD = "bid_plus_spread"
    MID_PLUS_SPREAD = "mid_plus_spread"


class SpreadUnit(StrEnum):
    """Unit used by a source spread field."""

    PRICE = "price"
    PIPS = "pips"
    POINTS = "points"


@dataclass(frozen=True, slots=True)
class TickCSVSchema:
    """Column map for two-sided tick files."""

    timestamp: str = "timestamp"
    bid: str = "bid"
    ask: str = "ask"
    symbol: str | None = "symbol"
    bid_size: str | None = "bid_size"
    ask_size: str | None = "ask_size"
    sequence: str | None = "sequence"
    received_time: str | None = None
    default_symbol: str | None = None
    timestamp_format: str | None = None
    naive_timezone: str = "UTC"


@dataclass(frozen=True, slots=True)
class BarCSVSchema:
    """Column map for OHLC files.

    Native ``BID_ASK`` bars are strongly preferred. The spread-derived modes are
    provided for vendors that publish only bid or mid bars. Deriving ask or both
    sides from one spread value is necessarily an approximation and should be
    identified in backtest metadata.
    """

    timestamp: str = "timestamp"
    symbol: str | None = "symbol"
    timeframe: str | None = "timeframe"
    default_symbol: str | None = None
    default_timeframe: Timeframe | None = None
    quote_mode: BarQuoteMode = BarQuoteMode.BID_ASK

    bid_open: str = "bid_open"
    bid_high: str = "bid_high"
    bid_low: str = "bid_low"
    bid_close: str = "bid_close"

    ask_open: str = "ask_open"
    ask_high: str = "ask_high"
    ask_low: str = "ask_low"
    ask_close: str = "ask_close"

    mid_open: str = "mid_open"
    mid_high: str = "mid_high"
    mid_low: str = "mid_low"
    mid_close: str = "mid_close"

    spread: str = "spread"
    spread_unit: SpreadUnit = SpreadUnit.PRICE
    tick_volume: str | None = "tick_volume"
    real_volume: str | None = "real_volume"
    complete: str | None = "complete"
    timestamp_format: str | None = None
    naive_timezone: str = "UTC"
