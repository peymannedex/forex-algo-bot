"""Read-only live market-data sources for paper-mode integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from fxbot.data.adapters.mt5_adapter import (
    MT5ConnectionConfig,
    MT5MarketDataAdapter,
)
from fxbot.domain.models import LiveSubscription, MarketDataRecord
from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.production.config import ProductionSettings


class LiveMarketRecordSource(Protocol):
    """Minimal async source contract consumed by the soak runner."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def stream(
        self,
        subscription: LiveSubscription,
    ) -> AsyncIterator[MarketDataRecord]: ...


class MT5ReadOnlyMarketSource:
    """MT5 market-data bridge that exposes no order-submission methods."""

    def __init__(
        self,
        production: ProductionSettings,
        live: PaperLiveFeedSettings,
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
        return self._adapter.stream(subscription)
