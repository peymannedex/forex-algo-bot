"""Read-only live market-data sources for paper-mode integration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

from fxbot.data.adapters.mt5_adapter import (
    MT5AdapterError,
    MT5ConnectionConfig,
    MT5MarketDataAdapter,
)
from fxbot.domain.models import LiveSubscription, MarketDataRecord, Tick
from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.production.config import ProductionSettings

Clock = Callable[[], datetime]


class LiveMarketRecordSource(Protocol):
    """Minimal async source contract consumed by the soak runner."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def stream(
        self,
        subscription: LiveSubscription,
    ) -> AsyncIterator[MarketDataRecord]: ...


class MT5ReadOnlyMarketSource:
    """Read-only MT5 bridge with configurable server-time normalization."""

    def __init__(
        self,
        production: ProductionSettings,
        live: PaperLiveFeedSettings,
        *,
        clock: Clock | None = None,
    ) -> None:
        password = (
            production.mt5_password.get_secret_value()
            if production.mt5_password is not None
            else None
        )
        self._adapter = MT5MarketDataAdapter(
            connection=MT5ConnectionConfig(
                terminal_path=production.mt5_terminal_path,
                login=production.mt5_login,
                password=password,
                server=production.mt5_server,
            ),
            poll_interval_seconds=live.poll_interval_seconds,
            emit_incomplete_bars=False,
            source="mt5-readonly-paper",
        )
        self._server_utc_offset = timedelta(
            minutes=live.mt5_server_utc_offset_minutes
        )
        self._max_future_skew = timedelta(
            seconds=live.max_future_skew_seconds
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def connected(self) -> bool:
        return self._adapter.connected

    async def connect(self) -> None:
        await self._adapter.connect()

    async def disconnect(self) -> None:
        await self._adapter.disconnect()

    def stream(
        self,
        subscription: LiveSubscription,
    ) -> AsyncIterator[MarketDataRecord]:
        return self._stream_normalized(subscription)

    async def _stream_normalized(
        self,
        subscription: LiveSubscription,
    ) -> AsyncIterator[MarketDataRecord]:
        async for record in self._adapter.stream(subscription):
            normalized = self._normalize_record(record)
            self._validate_live_timestamp(normalized)
            yield normalized

    def _normalize_record(
        self,
        record: MarketDataRecord,
    ) -> MarketDataRecord:
        if not self._server_utc_offset:
            return record
        if isinstance(record, Tick):
            return replace(
                record,
                event_time=record.event_time - self._server_utc_offset,
            )
        return replace(
            record,
            open_time=record.open_time - self._server_utc_offset,
        )

    def _validate_live_timestamp(
        self,
        record: MarketDataRecord,
    ) -> None:
        timestamp = (
            record.event_time
            if isinstance(record, Tick)
            else record.close_time
        )
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware timestamp")
        now_utc = now.astimezone(UTC)
        future_limit = now_utc + self._max_future_skew
        if timestamp > future_limit:
            lead_seconds = (timestamp - now_utc).total_seconds()
            raise MT5AdapterError(
                "MT5 record remains future-dated after UTC normalization: "
                f"{lead_seconds:.3f}s ahead. Check "
                "FXBOT_PAPER_LIVE_MT5_SERVER_UTC_OFFSET_MINUTES."
            )
