"""Abstract contracts for historical and live market-data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from fxbot.domain.models import Bar, HistoricalDataRequest, LiveSubscription, MarketDataRecord, Tick


class MarketDataAdapterError(RuntimeError):
    """Base exception for adapter, transport, and source-format failures."""


@dataclass(frozen=True, slots=True)
class AdapterDiagnostics:
    """Snapshot of source-row parsing outcomes."""

    rows_read: int = 0
    records_emitted: int = 0
    records_rejected: int = 0
    errors: tuple[str, ...] = ()


class HistoricalMarketDataAdapter(ABC):
    """Synchronous streaming interface for local or remote historical data."""

    @property
    @abstractmethod
    def diagnostics(self) -> AdapterDiagnostics:
        """Return diagnostics for the most recent iterator invocation."""

    @abstractmethod
    def iter_ticks(self, request: HistoricalDataRequest) -> Iterator[Tick]:
        """Yield normalized ticks matching the request."""

    @abstractmethod
    def iter_bars(self, request: HistoricalDataRequest) -> Iterator[Bar]:
        """Yield normalized bars matching the request."""


class LiveMarketDataAdapter(ABC):
    """Asynchronous source of normalized live market-data records."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the transport is ready to stream records."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the underlying transport."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the transport and unblock active consumers."""

    @abstractmethod
    def stream(self, subscription: LiveSubscription) -> AsyncIterator[MarketDataRecord]:
        """Return an asynchronous iterator of accepted records."""

    async def __aenter__(self) -> LiveMarketDataAdapter:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()
