"""MetaTrader 5 market-data adapter.

The official ``MetaTrader5`` Python package is a synchronous bridge to a local
MT5 terminal.  This adapter keeps that platform-specific API behind the
canonical Phase 1 data contracts and uses ``asyncio.to_thread`` for live polling
so the event loop is never blocked by terminal calls.

Important limitations:

* The MetaTrader5 wheel and terminal integration are Windows-specific.
* MT5 historical OHLC rates are bid bars plus a spread value in *points*.
  Ask OHLC values are therefore reconstructed with a constant bar spread and
  marked through the source string.  Tick resampling remains preferable when
  exact bid/ask/mid bar alignment is required.
* MT5 does not provide push callbacks through the Python package.  Live data is
  polled and deduplicated.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from fxbot.data.adapters.base import (
    AdapterDiagnostics,
    HistoricalMarketDataAdapter,
    LiveMarketDataAdapter,
    MarketDataAdapterError,
)
from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import (
    OHLC,
    Bar,
    HistoricalDataRequest,
    LiveSubscription,
    MarketDataRecord,
    SymbolSpec,
    Tick,
)


class MT5AdapterError(MarketDataAdapterError):
    """Raised when the MT5 terminal or Python bridge rejects an operation."""


@runtime_checkable
class MT5ClientProtocol(Protocol):
    """Minimal structural contract used by the adapter and unit-test fakes."""

    COPY_TICKS_ALL: int

    def initialize(self, *args: Any, **kwargs: Any) -> bool: ...

    def shutdown(self) -> None: ...

    def last_error(self) -> Any: ...

    def symbol_select(self, symbol: str, enable: bool) -> bool: ...

    def symbol_info(self, symbol: str) -> Any: ...

    def symbol_info_tick(self, symbol: str) -> Any: ...

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
        flags: int,
    ) -> Any: ...

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime,
    ) -> Any: ...

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: int,
        start_pos: int,
        count: int,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class MT5ConnectionConfig:
    """Connection parameters forwarded to ``MetaTrader5.initialize``.

    Passwords are intentionally not represented in ``repr`` output.
    Prefer environment-variable or secret-manager injection rather than
    checking credentials into configuration files.
    """

    terminal_path: Path | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 60_000
    portable: bool = False

    def __post_init__(self) -> None:
        if self.login is not None and self.login <= 0:
            raise ValueError("login must be positive")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    def initialize_args(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        positional: tuple[Any, ...] = (
            (str(self.terminal_path),) if self.terminal_path is not None else ()
        )
        keyword: dict[str, Any] = {
            "timeout": self.timeout_ms,
            "portable": self.portable,
        }
        if self.login is not None:
            keyword["login"] = self.login
        if self.password is not None:
            keyword["password"] = self.password
        if self.server is not None:
            keyword["server"] = self.server
        return positional, keyword

    def __repr__(self) -> str:
        return (
            "MT5ConnectionConfig("
            f"terminal_path={self.terminal_path!r}, login={self.login!r}, "
            f"password={'***' if self.password is not None else None!r}, "
            f"server={self.server!r}, timeout_ms={self.timeout_ms!r}, "
            f"portable={self.portable!r})"
        )


@dataclass(slots=True)
class _MutableDiagnostics:
    rows_read: int = 0
    records_emitted: int = 0
    records_rejected: int = 0
    errors: list[str] | None = None

    def snapshot(self) -> AdapterDiagnostics:
        return AdapterDiagnostics(
            rows_read=self.rows_read,
            records_emitted=self.records_emitted,
            records_rejected=self.records_rejected,
            errors=tuple(self.errors or ()),
        )


class MT5MarketDataAdapter(HistoricalMarketDataAdapter, LiveMarketDataAdapter):
    """Historical and live market-data adapter backed by a local MT5 terminal.

    Historical methods connect lazily if necessary.  For lifecycle clarity,
    production code should normally use ``async with adapter`` for live
    streams and call :meth:`close` after synchronous historical extraction.
    """

    def __init__(
        self,
        *,
        connection: MT5ConnectionConfig | None = None,
        poll_interval_seconds: float = 0.10,
        emit_incomplete_bars: bool = True,
        client: MT5ClientProtocol | None = None,
        source: str = "mt5",
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._connection = connection or MT5ConnectionConfig()
        self._poll_interval_seconds = poll_interval_seconds
        self._emit_incomplete_bars = emit_incomplete_bars
        self._client = client
        self._source = source.strip() or "mt5"
        self._connected = False
        self._diagnostics = _MutableDiagnostics(errors=[])
        self._stop_event: asyncio.Event | None = None
        self._last_tick_signature: dict[str, tuple[int, float, float]] = {}
        self._last_bar_signature: dict[
            tuple[str, Timeframe, datetime], tuple[float, float, float, float, int]
        ] = {}

    @property
    def diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics.snapshot()

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        await asyncio.to_thread(self._connect_blocking)
        self._stop_event = asyncio.Event()

    async def disconnect(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if not self._connected:
            return
        await asyncio.to_thread(self.close)

    def close(self) -> None:
        """Synchronously shut down the MT5 bridge if this adapter connected it."""

        if self._connected:
            self._require_client().shutdown()
            self._connected = False

    def iter_ticks(self, request: HistoricalDataRequest) -> Iterator[Tick]:
        if request.kind is not DataKind.TICK:
            raise ValueError("iter_ticks requires a tick HistoricalDataRequest")
        start, end = self._require_bounded_request(request)
        self._ensure_connected()
        self._select_symbol(request.symbol)
        self._diagnostics = _MutableDiagnostics(errors=[])

        client = self._require_client()
        raw = client.copy_ticks_range(
            request.symbol,
            start,
            end,
            client.COPY_TICKS_ALL,
        )
        if raw is None:
            raise self._operation_error("copy_ticks_range")

        for row in raw:
            self._diagnostics.rows_read += 1
            try:
                tick = self._tick_from_row(request.symbol, row, live=False)
                if request.contains(tick.event_time):
                    self._diagnostics.records_emitted += 1
                    yield tick
            except (KeyError, TypeError, ValueError) as exc:
                self._reject_row("tick", exc)

    def iter_bars(self, request: HistoricalDataRequest) -> Iterator[Bar]:
        if request.kind is not DataKind.BAR:
            raise ValueError("iter_bars requires a bar HistoricalDataRequest")
        start, end = self._require_bounded_request(request)
        assert request.timeframe is not None
        self._ensure_connected()
        self._select_symbol(request.symbol)
        self._diagnostics = _MutableDiagnostics(errors=[])

        client = self._require_client()
        raw = client.copy_rates_range(
            request.symbol,
            self._mt5_timeframe(request.timeframe),
            start,
            end,
        )
        if raw is None:
            raise self._operation_error("copy_rates_range")

        for row in raw:
            self._diagnostics.rows_read += 1
            try:
                bar = self._bar_from_row(
                    request.symbol,
                    request.timeframe,
                    row,
                    complete=True,
                )
                if request.contains(bar.open_time):
                    self._diagnostics.records_emitted += 1
                    yield bar
            except (KeyError, TypeError, ValueError) as exc:
                self._reject_row("bar", exc)

    def stream(self, subscription: LiveSubscription) -> AsyncIterator[MarketDataRecord]:
        return self._stream_records(subscription)

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        """Build a canonical :class:`SymbolSpec` from MT5 symbol metadata."""

        self._ensure_connected()
        self._select_symbol(symbol)
        info = self._require_client().symbol_info(symbol)
        if info is None:
            raise self._operation_error(f"symbol_info({symbol})")

        normalized = symbol.strip().upper()
        base = str(_field(info, "currency_base", normalized[:3])).upper()
        quote = str(_field(info, "currency_profit", normalized[3:6])).upper()
        digits = int(_field(info, "digits"))
        point = float(_field(info, "point"))
        contract_size = float(_field(info, "trade_contract_size", 100_000.0))
        pip_size = point * 10.0 if digits in (3, 5) else point
        return SymbolSpec(
            symbol=normalized,
            base_currency=base,
            quote_currency=quote,
            digits=digits,
            point_size=point,
            pip_size=pip_size,
            contract_size=contract_size,
        )

    async def _stream_records(
        self,
        subscription: LiveSubscription,
    ) -> AsyncIterator[MarketDataRecord]:
        if not self._connected:
            raise MT5AdapterError("Adapter must be connected before streaming")
        for symbol in sorted(subscription.symbols):
            await asyncio.to_thread(self._select_symbol, symbol)

        stop_event = self._stop_event or asyncio.Event()
        while self._connected and not stop_event.is_set():
            records = await asyncio.to_thread(self._poll_once, subscription)
            for record in records:
                yield record
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )

    def _poll_once(self, subscription: LiveSubscription) -> tuple[MarketDataRecord, ...]:
        records: list[MarketDataRecord] = []
        client = self._require_client()

        if Timeframe.TICK in subscription.timeframes:
            for symbol in sorted(subscription.symbols):
                raw_tick = client.symbol_info_tick(symbol)
                if raw_tick is None:
                    continue
                tick = self._tick_from_row(symbol, raw_tick, live=True)
                fallback_msc = int(tick.event_time.timestamp() * 1000)
                time_msc = int(_field(raw_tick, "time_msc", fallback_msc))
                tick_sig = (time_msc, tick.bid, tick.ask)
                if self._last_tick_signature.get(symbol) != tick_sig:
                    self._last_tick_signature[symbol] = tick_sig
                    records.append(tick)

        bar_timeframes = sorted(
            (item for item in subscription.timeframes if item is not Timeframe.TICK),
            key=lambda item: item.seconds or 0,
        )
        for symbol in sorted(subscription.symbols):
            for timeframe in bar_timeframes:
                raw_rates = client.copy_rates_from_pos(
                    symbol,
                    self._mt5_timeframe(timeframe),
                    0,
                    2,
                )
                if raw_rates is None:
                    continue
                sorted_rows = sorted(raw_rates, key=lambda row: int(_field(row, "time")))
                for index, row in enumerate(sorted_rows):
                    is_current = index == len(sorted_rows) - 1
                    if is_current and not self._emit_incomplete_bars:
                        continue
                    bar = self._bar_from_row(
                        symbol,
                        timeframe,
                        row,
                        complete=not is_current,
                    )
                    bar_sig = (
                        bar.bid.open,
                        bar.bid.high,
                        bar.bid.low,
                        bar.bid.close,
                        bar.tick_volume,
                    )
                    key = (symbol, timeframe, bar.open_time)
                    if self._last_bar_signature.get(key) != bar_sig:
                        self._last_bar_signature[key] = bar_sig
                        records.append(bar)

        records.sort(key=_market_record_time)
        return tuple(records)

    def _connect_blocking(self) -> None:
        if self._connected:
            return
        client = self._require_client()
        args, kwargs = self._connection.initialize_args()
        if not client.initialize(*args, **kwargs):
            raise self._operation_error("initialize")
        self._connected = True

    def _ensure_connected(self) -> None:
        if not self._connected:
            self._connect_blocking()

    def _select_symbol(self, symbol: str) -> None:
        if not self._require_client().symbol_select(symbol, True):
            raise self._operation_error(f"symbol_select({symbol})")

    def _tick_from_row(self, symbol: str, row: Any, *, live: bool) -> Tick:
        time_msc = _field(row, "time_msc", None)
        event_time = (
            datetime.fromtimestamp(float(time_msc) / 1000.0, tz=UTC)
            if time_msc is not None
            else datetime.fromtimestamp(float(_field(row, "time")), tz=UTC)
        )
        return Tick(
            symbol=symbol,
            event_time=event_time,
            bid=float(_field(row, "bid")),
            ask=float(_field(row, "ask")),
            source=self._source,
            received_time=datetime.now(UTC) if live else None,
        )

    def _bar_from_row(
        self,
        symbol: str,
        timeframe: Timeframe,
        row: Any,
        *,
        complete: bool,
    ) -> Bar:
        info = self._require_client().symbol_info(symbol)
        if info is None:
            raise self._operation_error(f"symbol_info({symbol})")
        point = float(_field(info, "point"))
        spread_price = float(_field(row, "spread", 0.0)) * point
        bid = OHLC(
            open=float(_field(row, "open")),
            high=float(_field(row, "high")),
            low=float(_field(row, "low")),
            close=float(_field(row, "close")),
        )
        ask = OHLC(
            open=bid.open + spread_price,
            high=bid.high + spread_price,
            low=bid.low + spread_price,
            close=bid.close + spread_price,
        )
        return Bar(
            symbol=symbol,
            open_time=datetime.fromtimestamp(float(_field(row, "time")), tz=UTC),
            timeframe=timeframe,
            bid=bid,
            ask=ask,
            tick_volume=int(_field(row, "tick_volume", 0)),
            real_volume=float(_field(row, "real_volume", 0.0)),
            source=f"{self._source}:bid-plus-spread",
            complete=complete,
        )

    def _mt5_timeframe(self, timeframe: Timeframe) -> int:
        names = {
            Timeframe.M1: "TIMEFRAME_M1",
            Timeframe.M5: "TIMEFRAME_M5",
            Timeframe.M15: "TIMEFRAME_M15",
            Timeframe.M30: "TIMEFRAME_M30",
            Timeframe.H1: "TIMEFRAME_H1",
            Timeframe.H4: "TIMEFRAME_H4",
            Timeframe.D1: "TIMEFRAME_D1",
            Timeframe.W1: "TIMEFRAME_W1",
        }
        try:
            name = names[timeframe]
        except KeyError as exc:
            raise MT5AdapterError(
                f"MT5 native bars do not support {timeframe.value}; stream ticks and resample"
            ) from exc
        try:
            return int(getattr(self._require_client(), name))
        except AttributeError as exc:
            raise MT5AdapterError(f"MT5 client does not expose {name}") from exc

    def _require_client(self) -> MT5ClientProtocol:
        if self._client is None:
            try:
                module: ModuleType = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise MT5AdapterError(
                    "MetaTrader5 is not installed. Install it on the Windows host "
                    "running the MT5 terminal."
                ) from exc
            self._client = module
        return self._client

    @staticmethod
    def _require_bounded_request(
        request: HistoricalDataRequest,
    ) -> tuple[datetime, datetime]:
        if request.start is None or request.end is None:
            raise ValueError("MT5 historical requests require both start and end")
        return request.start, request.end

    def _operation_error(self, operation: str) -> MT5AdapterError:
        error = self._require_client().last_error()
        return MT5AdapterError(f"MT5 {operation} failed: {error!r}")

    def _reject_row(self, record_type: str, exc: Exception) -> None:
        self._diagnostics.records_rejected += 1
        if self._diagnostics.errors is not None and len(self._diagnostics.errors) < 100:
            self._diagnostics.errors.append(f"Invalid MT5 {record_type}: {exc}")


def _field(row: Any, name: str, default: Any = ...) -> Any:
    """Read a field from dicts, namedtuples, NumPy records, or simple fakes."""

    if isinstance(row, dict):
        if name in row:
            return row[name]
    else:
        try:
            return getattr(row, name)
        except AttributeError:
            pass
        try:
            return row[name]
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    if default is not ...:
        return default
    raise KeyError(name)


def _market_record_time(record: MarketDataRecord) -> datetime:
    return record.event_time if isinstance(record, Tick) else record.open_time
