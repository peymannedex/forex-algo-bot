from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import cast

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import LiveSubscription, MarketDataRecord
from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.integration.models import PaperCycleResult, PaperFrame
from fxbot.integration.runtime import PaperIntegrationRuntime
from fxbot.integration.soak import (
    LivePaperFrameAssembler,
    PaperLiveSoakRunner,
    SoakEvidenceWriter,
)
from fxbot.production.health import HealthRegistry


class SilentSource:
    def __init__(self) -> None:
        self.connects = 0
        self.disconnects = 0
        self.cancelled = False

    async def connect(self) -> None:
        self.connects += 1

    async def disconnect(self) -> None:
        self.disconnects += 1

    async def _records(self) -> AsyncIterator[MarketDataRecord]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if False:
            yield cast(MarketDataRecord, object())

    def stream(
        self,
        subscription: LiveSubscription,
    ) -> AsyncIterator[MarketDataRecord]:
        del subscription
        return self._records()


class RuntimeStub:
    last_frame_at: datetime | None = None

    def __init__(self) -> None:
        self.stopped = False

    def process(self, frame: PaperFrame) -> PaperCycleResult:
        raise AssertionError(f"Unexpected frame: {frame!r}")

    def stop(self) -> None:
        self.stopped = True


def build_runner(
    tmp_path: Path,
) -> tuple[PaperLiveSoakRunner, SilentSource, RuntimeStub]:
    source = SilentSource()
    runtime = RuntimeStub()
    health = HealthRegistry()
    settings = PaperLiveFeedSettings(
        poll_interval_seconds=0.01,
        reconnect_delay_seconds=0.01,
        evidence_directory=tmp_path,
    )
    evidence = SoakEvidenceWriter(tmp_path, health=health)
    runner = PaperLiveSoakRunner(
        source=source,
        runtime=cast(PaperIntegrationRuntime, runtime),
        health=health,
        subscription=LiveSubscription(
            symbols=frozenset({"EURUSD"}),
            timeframes=frozenset({Timeframe.M5, Timeframe.M15}),
        ),
        assembler=LivePaperFrameAssembler(
            primary_timeframe=Timeframe.M5,
            required_timeframes=(Timeframe.M5, Timeframe.M15),
            history_limit=100,
        ),
        evidence=evidence,
        settings=settings,
        stop_file=tmp_path / "STOP",
    )
    return runner, source, runtime


def test_max_seconds_stops_silent_stream(tmp_path: Path) -> None:
    async def scenario():
        runner, source, runtime = build_runner(tmp_path)
        summary = await asyncio.wait_for(
            runner.run(asyncio.Event(), max_seconds=0.05),
            timeout=1.0,
        )
        return summary, source, runtime

    summary, source, runtime = asyncio.run(scenario())

    assert summary.cycles == 0
    assert source.connects == 1
    assert source.disconnects == 1
    assert source.cancelled
    assert runtime.stopped


def test_stop_event_interrupts_silent_stream(tmp_path: Path) -> None:
    async def scenario():
        runner, source, runtime = build_runner(tmp_path)
        stop_event = asyncio.Event()
        task = asyncio.create_task(runner.run(stop_event))
        await asyncio.sleep(0.03)
        stop_event.set()
        summary = await asyncio.wait_for(task, timeout=1.0)
        return summary, source, runtime

    summary, source, runtime = asyncio.run(scenario())

    assert summary.cycles == 0
    assert source.disconnects == 1
    assert source.cancelled
    assert runtime.stopped


def test_stop_file_interrupts_silent_stream(tmp_path: Path) -> None:
    async def scenario():
        runner, source, runtime = build_runner(tmp_path)
        task = asyncio.create_task(runner.run(asyncio.Event()))
        await asyncio.sleep(0.03)
        (tmp_path / "STOP").touch()
        summary = await asyncio.wait_for(task, timeout=1.0)
        return summary, source, runtime

    summary, source, runtime = asyncio.run(scenario())

    assert summary.cycles == 0
    assert source.disconnects == 1
    assert source.cancelled
    assert runtime.stopped
