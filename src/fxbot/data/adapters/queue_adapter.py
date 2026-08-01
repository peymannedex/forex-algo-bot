"""Bounded asynchronous live-data adapter and broker-integration boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fxbot.data.adapters.base import LiveMarketDataAdapter, MarketDataAdapterError
from fxbot.domain.enums import QueueOverflowPolicy
from fxbot.domain.models import LiveSubscription, MarketDataRecord

_STOP = object()


class AsyncQueueLiveDataAdapter(LiveMarketDataAdapter):
    """Transport-neutral live adapter backed by a bounded ``asyncio.Queue``.

    A future MT5, cTrader, REST, WebSocket, or FIX connector can parse its native
    messages and call :meth:`publish`. Downstream strategy code consumes only the
    normalized :class:`Tick` and :class:`Bar` models.

    This adapter intentionally has one consumer stream. Fan-out belongs in the
    event bus, which is a later module, so data ingestion remains independent of
    strategy count.
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 100_000,
        overflow_policy: QueueOverflowPolicy = QueueOverflowPolicy.BLOCK,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._queue: asyncio.Queue[MarketDataRecord | object] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._overflow_policy = QueueOverflowPolicy(overflow_policy)
        self._connected = False
        self._consumer_active = False
        self.dropped_records = 0

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        # A disconnected feed must not replay stale quotes or a previous stop
        # marker after reconnection. Freshness is more important than retention
        # at this transport boundary.
        while True:
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
                if item is not _STOP:
                    self.dropped_records += 1
            except asyncio.QueueEmpty:
                break
        self._connected = True

    async def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        await self._force_put(_STOP)

    async def publish(self, record: MarketDataRecord) -> None:
        """Publish one validated record from a broker/feed producer."""

        if not self._connected:
            raise MarketDataAdapterError("Cannot publish while adapter is disconnected")

        if self._overflow_policy is QueueOverflowPolicy.BLOCK:
            await self._queue.put(record)
            return

        if not self._queue.full():
            self._queue.put_nowait(record)
            return

        if self._overflow_policy is QueueOverflowPolicy.RAISE:
            raise MarketDataAdapterError("Live market-data queue is full")

        # DROP_OLDEST prioritizes fresh quotes and records the loss explicitly.
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except asyncio.QueueEmpty:  # pragma: no cover - race protection
            pass
        self.dropped_records += 1
        self._queue.put_nowait(record)

    async def stream(self, subscription: LiveSubscription) -> AsyncIterator[MarketDataRecord]:
        if not self._connected:
            raise MarketDataAdapterError("Adapter must be connected before streaming")
        if self._consumer_active:
            raise MarketDataAdapterError("Only one active consumer stream is supported")

        self._consumer_active = True
        try:
            while True:
                item = await self._queue.get()
                self._queue.task_done()
                if item is _STOP:
                    break
                if subscription.accepts(item):  # type: ignore[arg-type]
                    yield item  # type: ignore[misc]
        finally:
            self._consumer_active = False

    async def _force_put(self, item: object) -> None:
        while self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.dropped_records += 1
            except asyncio.QueueEmpty:  # pragma: no cover - race protection
                break
        self._queue.put_nowait(item)
