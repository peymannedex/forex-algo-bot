from datetime import UTC, datetime

import pytest

from fxbot.data.adapters.queue_adapter import AsyncQueueLiveDataAdapter
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import LiveSubscription, Tick


@pytest.mark.asyncio
async def test_queue_live_adapter_streams_matching_record() -> None:
    adapter = AsyncQueueLiveDataAdapter(max_queue_size=10)
    subscription = LiveSubscription(
        symbols=frozenset({"EURUSD"}),
        timeframes=frozenset({Timeframe.TICK}),
    )
    tick = Tick("EURUSD", datetime.now(UTC), 1.1, 1.1001)

    await adapter.connect()
    await adapter.publish(tick)
    stream = adapter.stream(subscription)
    received = await anext(stream)
    await stream.aclose()
    await adapter.disconnect()

    assert received == tick
