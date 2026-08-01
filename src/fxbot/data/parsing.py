"""Row-to-domain parsing shared by CSV and Parquet adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fxbot.data.schemas import BarCSVSchema, BarQuoteMode, SpreadUnit, TickCSVSchema
from fxbot.data.time_utils import parse_timestamp
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar, SymbolSpec, Tick


class RowParseError(ValueError):
    """Raised when a source row cannot be converted into a domain model."""


def _required(row: Mapping[str, Any], column: str) -> Any:
    try:
        value = row[column]
    except KeyError as exc:
        raise RowParseError(f"Missing required column {column!r}") from exc
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RowParseError(f"Required column {column!r} is empty")
    return value


def _optional(row: Mapping[str, Any], column: str | None) -> Any | None:
    if column is None:
        return None
    value = row.get(column)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


def _float(row: Mapping[str, Any], column: str) -> float:
    value = _required(row, column)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RowParseError(f"Column {column!r} is not numeric: {value!r}") from exc


def _optional_float(row: Mapping[str, Any], column: str | None) -> float | None:
    value = _optional(row, column)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RowParseError(f"Column {column!r} is not numeric: {value!r}") from exc


def _optional_int(row: Mapping[str, Any], column: str | None, default: int = 0) -> int:
    value = _optional(row, column)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RowParseError(f"Column {column!r} is not an integer: {value!r}") from exc


def _bool(row: Mapping[str, Any], column: str | None, default: bool = True) -> bool:
    value = _optional(row, column)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "complete"}:
        return True
    if normalized in {"0", "false", "no", "n", "incomplete"}:
        return False
    raise RowParseError(f"Column {column!r} is not boolean-like: {value!r}")


def _symbol(row: Mapping[str, Any], column: str | None, default: str | None) -> str:
    value = _optional(row, column)
    result = str(value).strip() if value is not None else default
    if result is None or not result.strip():
        raise RowParseError("No symbol column value or default_symbol was provided")
    return result


class MarketDataRowParser:
    """Convert raw source mappings into validated immutable domain objects."""

    def __init__(self, symbol_specs: Mapping[str, SymbolSpec] | None = None) -> None:
        self._symbol_specs = {key.upper(): value for key, value in (symbol_specs or {}).items()}

    def parse_tick(
        self,
        row: Mapping[str, Any],
        schema: TickCSVSchema,
        *,
        source: str,
    ) -> Tick:
        try:
            symbol = _symbol(row, schema.symbol, schema.default_symbol)
            received = _optional(row, schema.received_time)
            return Tick(
                symbol=symbol,
                event_time=parse_timestamp(
                    _required(row, schema.timestamp),
                    timestamp_format=schema.timestamp_format,
                    naive_timezone=schema.naive_timezone,
                ),
                bid=_float(row, schema.bid),
                ask=_float(row, schema.ask),
                bid_size=_optional_float(row, schema.bid_size),
                ask_size=_optional_float(row, schema.ask_size),
                sequence=(
                    _optional_int(row, schema.sequence)
                    if _optional(row, schema.sequence) is not None
                    else None
                ),
                received_time=(
                    parse_timestamp(
                        received,
                        timestamp_format=schema.timestamp_format,
                        naive_timezone=schema.naive_timezone,
                    )
                    if received is not None
                    else None
                ),
                source=source,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, RowParseError):
                raise
            raise RowParseError(str(exc)) from exc

    def parse_bar(
        self,
        row: Mapping[str, Any],
        schema: BarCSVSchema,
        *,
        source: str,
    ) -> Bar:
        try:
            symbol = _symbol(row, schema.symbol, schema.default_symbol)
            timeframe_value = _optional(row, schema.timeframe)
            timeframe_source: str | Timeframe | None = (
                str(timeframe_value)
                if timeframe_value is not None
                else schema.default_timeframe
            )
            if timeframe_source is None:
                raise RowParseError(
                    "No timeframe column value or default_timeframe was provided"
                )
            timeframe = Timeframe.parse(timeframe_source)
            bid, ask = self._parse_bar_quotes(row, schema, symbol)
            return Bar(
                symbol=symbol,
                open_time=parse_timestamp(
                    _required(row, schema.timestamp),
                    timestamp_format=schema.timestamp_format,
                    naive_timezone=schema.naive_timezone,
                ),
                timeframe=timeframe,
                bid=bid,
                ask=ask,
                tick_volume=_optional_int(row, schema.tick_volume),
                real_volume=_optional_float(row, schema.real_volume),
                complete=_bool(row, schema.complete),
                source=source,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, RowParseError):
                raise
            raise RowParseError(str(exc)) from exc

    def _parse_bar_quotes(
        self,
        row: Mapping[str, Any],
        schema: BarCSVSchema,
        symbol: str,
    ) -> tuple[OHLC, OHLC]:
        mode = BarQuoteMode(schema.quote_mode)
        if mode is BarQuoteMode.BID_ASK:
            return (
                OHLC(
                    open=_float(row, schema.bid_open),
                    high=_float(row, schema.bid_high),
                    low=_float(row, schema.bid_low),
                    close=_float(row, schema.bid_close),
                ),
                OHLC(
                    open=_float(row, schema.ask_open),
                    high=_float(row, schema.ask_high),
                    low=_float(row, schema.ask_low),
                    close=_float(row, schema.ask_close),
                ),
            )

        spread = self._spread_in_price(row, schema, symbol)
        if mode is BarQuoteMode.BID_PLUS_SPREAD:
            bid = OHLC(
                open=_float(row, schema.bid_open),
                high=_float(row, schema.bid_high),
                low=_float(row, schema.bid_low),
                close=_float(row, schema.bid_close),
            )
            ask = OHLC(
                open=bid.open + spread,
                high=bid.high + spread,
                low=bid.low + spread,
                close=bid.close + spread,
            )
            return bid, ask

        mid = OHLC(
            open=_float(row, schema.mid_open),
            high=_float(row, schema.mid_high),
            low=_float(row, schema.mid_low),
            close=_float(row, schema.mid_close),
        )
        half = spread / 2.0
        return (
            OHLC(
                open=mid.open - half,
                high=mid.high - half,
                low=mid.low - half,
                close=mid.close - half,
            ),
            OHLC(
                open=mid.open + half,
                high=mid.high + half,
                low=mid.low + half,
                close=mid.close + half,
            ),
        )

    def _spread_in_price(
        self,
        row: Mapping[str, Any],
        schema: BarCSVSchema,
        symbol: str,
    ) -> float:
        spread = _float(row, schema.spread)
        unit = SpreadUnit(schema.spread_unit)
        if unit is SpreadUnit.PRICE:
            return spread

        try:
            spec = self._symbol_specs[symbol.upper()]
        except KeyError as exc:
            raise RowParseError(
                f"SymbolSpec for {symbol!r} is required to convert {unit.value} spread"
            ) from exc
        return spread * (spec.pip_size if unit is SpreadUnit.PIPS else spec.point_size)
