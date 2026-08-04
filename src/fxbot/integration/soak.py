"""Sustained paper-mode live-feed runner and acceptance evidence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import Bar, LiveSubscription, MarketDataRecord
from fxbot.execution.models import Quote
from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.integration.live_feed import LiveMarketRecordSource
from fxbot.integration.models import PaperCycleResult, PaperFrame
from fxbot.integration.runtime import PaperIntegrationRuntime
from fxbot.production.health import ComponentState, HealthRegistry
from fxbot.strategy.context import MarketContextBuilder

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class SoakRunSummary:
    """Serializable aggregate for one service process."""

    started_at: datetime
    updated_at: datetime
    cycles: int
    accepted_orders: int
    rejected_orders: int
    fills: int
    errors: int
    decision_counts: tuple[tuple[str, int], ...]
    ending_balance: float | None
    ending_equity: float | None
    last_frame_at: datetime | None
    health: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "cycles": self.cycles,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "fills": self.fills,
            "errors": self.errors,
            "decision_counts": dict(self.decision_counts),
            "ending_balance": self.ending_balance,
            "ending_equity": self.ending_equity,
            "last_frame_at": (
                self.last_frame_at.isoformat()
                if self.last_frame_at is not None
                else None
            ),
            "health": self.health,
        }


class LivePaperFrameAssembler:
    """Convert completed live bars into look-ahead-safe paper frames."""

    def __init__(
        self,
        *,
        primary_timeframe: Timeframe,
        required_timeframes: tuple[Timeframe, ...],
        history_limit: int,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.primary_timeframe = Timeframe.parse(primary_timeframe)
        self.required_timeframes = frozenset(
            Timeframe.parse(item) for item in required_timeframes
        )
        self.history_limit = history_limit
        self._builder = MarketContextBuilder()
        self._history: dict[
            tuple[str, Timeframe],
            deque[Bar],
        ] = defaultdict(lambda: deque(maxlen=self.history_limit))
        self._seen: set[tuple[str, Timeframe, datetime]] = set()

    def ingest(self, record: MarketDataRecord) -> PaperFrame | None:
        if not isinstance(record, Bar):
            return None
        if not record.complete:
            return None
        if record.timeframe not in self.required_timeframes:
            return None

        identity = (record.symbol, record.timeframe, record.open_time)
        if identity in self._seen:
            return None
        self._seen.add(identity)
        self._history[(record.symbol, record.timeframe)].append(record)

        if record.timeframe is not self.primary_timeframe:
            return None

        as_of = record.close_time
        bars = tuple(
            sorted(
                (
                    bar
                    for (symbol, _), history in self._history.items()
                    if symbol == record.symbol
                    for bar in history
                    if bar.close_time <= as_of
                ),
                key=lambda item: (
                    item.close_time,
                    item.timeframe.value,
                ),
            )
        )
        context = self._builder.build(
            symbol=record.symbol,
            as_of=as_of,
            primary_timeframe=self.primary_timeframe,
            bars=bars,
        )
        return PaperFrame(
            context=context,
            quote=Quote(
                symbol=record.symbol,
                bid=record.bid.close,
                ask=record.ask.close,
                timestamp=as_of,
            ),
        )


class SoakEvidenceWriter:
    """Append cycle/error evidence and atomically maintain daily summaries."""

    def __init__(
        self,
        directory: Path,
        *,
        health: HealthRegistry,
        clock: Clock | None = None,
    ) -> None:
        self.directory = directory
        self.health = health
        self._clock = clock or (lambda: datetime.now(UTC))
        self.started_at = self._utc(self._clock())
        self._cycles = 0
        self._accepted_orders = 0
        self._rejected_orders = 0
        self._fills = 0
        self._errors = 0
        self._decisions: Counter[str] = Counter()
        self._ending_balance: float | None = None
        self._ending_equity: float | None = None
        self._last_frame_at: datetime | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cycle_path = self.directory / "cycles.jsonl"
        self._error_path = self.directory / "errors.jsonl"

    def record_cycle(self, result: PaperCycleResult) -> None:
        self._cycles += 1
        self._accepted_orders += result.accepted_orders
        self._rejected_orders += result.rejected_orders
        self._fills += result.sync.new_fills
        self._decisions[result.decision.action.value] += 1
        self._ending_balance = result.account.balance
        self._ending_equity = result.account.equity
        self._last_frame_at = result.processed_at
        self._append_json(
            self._cycle_path,
            {
                "cycle": result.cycle,
                "processed_at": result.processed_at.isoformat(),
                "decision": result.decision.action.value,
                "accepted_orders": result.accepted_orders,
                "rejected_orders": result.rejected_orders,
                "fills": result.sync.new_fills,
                "balance": result.account.balance,
                "equity": result.account.equity,
            },
        )
        self.write_summary()

    def record_error(self, exc: Exception) -> None:
        self._errors += 1
        self._append_json(
            self._error_path,
            {
                "timestamp": self._utc(self._clock()).isoformat(),
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        self.write_summary()

    def write_summary(self) -> SoakRunSummary:
        now = self._utc(self._clock())
        summary = SoakRunSummary(
            started_at=self.started_at,
            updated_at=now,
            cycles=self._cycles,
            accepted_orders=self._accepted_orders,
            rejected_orders=self._rejected_orders,
            fills=self._fills,
            errors=self._errors,
            decision_counts=tuple(sorted(self._decisions.items())),
            ending_balance=self._ending_balance,
            ending_equity=self._ending_equity,
            last_frame_at=self._last_frame_at,
            health=self.health.snapshot().to_dict(),
        )
        payload = json.dumps(
            summary.to_dict(),
            indent=2,
            sort_keys=True,
        ) + "\n"
        self._atomic_write(self.directory / "latest-summary.json", payload)
        self._atomic_write(
            self.directory / f"daily-{now.date().isoformat()}.json",
            payload,
        )
        return summary

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return timezone-aware timestamps")
        return value.astimezone(UTC)

    @staticmethod
    def _append_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)


class PaperLiveSoakRunner:
    """Reconnectable paper service driven only by live market data."""

    def __init__(
        self,
        *,
        source: LiveMarketRecordSource,
        runtime: PaperIntegrationRuntime,
        health: HealthRegistry,
        subscription: LiveSubscription,
        assembler: LivePaperFrameAssembler,
        evidence: SoakEvidenceWriter,
        settings: PaperLiveFeedSettings,
        stop_file: Path,
        clock: Clock | None = None,
    ) -> None:
        self.source = source
        self.runtime = runtime
        self.health = health
        self.subscription = subscription
        self.assembler = assembler
        self.evidence = evidence
        self.settings = settings
        self.stop_file = stop_file
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run(
        self,
        stop_event: asyncio.Event,
        *,
        max_cycles: int | None = None,
        max_seconds: float | None = None,
    ) -> SoakRunSummary:
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        if max_seconds is not None and max_seconds <= 0.0:
            raise ValueError("max_seconds must be positive")

        started = self._clock()
        consecutive_errors = 0
        processed_cycles = 0

        try:
            while not stop_event.is_set():
                if self.stop_file.exists():
                    stop_event.set()
                    break
                if self._seconds_since(started) >= (max_seconds or float("inf")):
                    stop_event.set()
                    break

                try:
                    await self.source.connect()
                    self.health.update(
                        "live_feed",
                        ComponentState.HEALTHY,
                        "read-only market-data source connected",
                    )

                    async for record in self.source.stream(self.subscription):
                        if stop_event.is_set() or self.stop_file.exists():
                            stop_event.set()
                            break

                        frame = self.assembler.ingest(record)
                        if frame is None:
                            continue
                        if (
                            self.runtime.last_frame_at is not None
                            and frame.quote.timestamp <= self.runtime.last_frame_at
                        ):
                            continue

                        result = self.runtime.process(frame)
                        self.evidence.record_cycle(result)
                        consecutive_errors = 0
                        processed_cycles += 1

                        if (
                            max_cycles is not None
                            and processed_cycles >= max_cycles
                        ):
                            stop_event.set()
                            break
                        if (
                            max_seconds is not None
                            and self._seconds_since(started) >= max_seconds
                        ):
                            stop_event.set()
                            break

                    if not stop_event.is_set():
                        raise RuntimeError("live market-data stream ended unexpectedly")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    consecutive_errors += 1
                    self.health.update(
                        "live_feed",
                        ComponentState.UNHEALTHY,
                        f"{type(exc).__name__}: {exc}",
                    )
                    self.evidence.record_error(exc)
                    if consecutive_errors >= self.settings.max_consecutive_errors:
                        raise
                    await asyncio.sleep(self.settings.reconnect_delay_seconds)
                finally:
                    with contextlib.suppress(Exception):
                        await self.source.disconnect()
        finally:
            self.runtime.stop()
            self.health.update(
                "service",
                ComponentState.STOPPED,
                "paper live-feed service stopped",
            )
            summary = self.evidence.write_summary()

        return summary

    def _seconds_since(self, started: datetime) -> float:
        return (self._clock() - started).total_seconds()
