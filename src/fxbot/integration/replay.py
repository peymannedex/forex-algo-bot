"""Chronological CSV replay source and paper acceptance summary."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.execution.models import Quote
from fxbot.integration.models import PaperCycleResult, PaperFrame
from fxbot.integration.runtime import PaperIntegrationRuntime
from fxbot.strategy.context import MarketContextBuilder


def _required(row: Mapping[str, str | None], name: str) -> str:
    value = row.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


_REQUIRED_COLUMNS = {
    "timestamp",
    "symbol",
    "timeframe",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
}


def load_replay_bars(path: Path) -> tuple[Bar, ...]:
    """Load completed bid/ask bars from a strict acceptance-test CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = _REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"replay CSV missing columns: {sorted(missing)}")
        bars: list[Bar] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(_required(row, "timestamp"))
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                bars.append(
                    Bar(
                        symbol=_required(row, "symbol"),
                        open_time=timestamp,
                        timeframe=Timeframe.parse(_required(row, "timeframe")),
                        bid=OHLC(
                            float(_required(row, "bid_open")),
                            float(_required(row, "bid_high")),
                            float(_required(row, "bid_low")),
                            float(_required(row, "bid_close")),
                        ),
                        ask=OHLC(
                            float(_required(row, "ask_open")),
                            float(_required(row, "ask_high")),
                            float(_required(row, "ask_low")),
                            float(_required(row, "ask_close")),
                        ),
                        tick_volume=int(row.get("tick_volume") or 0),
                        source="paper-replay",
                        complete=True,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid replay row {row_number}: {exc}") from exc
    bars.sort(key=lambda bar: (bar.close_time, bar.symbol, bar.timeframe.value))
    return tuple(bars)


def iter_paper_frames(
    bars: tuple[Bar, ...],
    *,
    primary_timeframe: Timeframe,
) -> tuple[PaperFrame, ...]:
    """Build look-ahead-safe strategy frames at each primary-bar close."""

    primary = Timeframe.parse(primary_timeframe)
    builder = MarketContextBuilder()
    history: list[Bar] = []
    frames: list[PaperFrame] = []
    for bar in bars:
        history.append(bar)
        if bar.timeframe is not primary:
            continue
        available = tuple(
            item
            for item in history
            if item.symbol == bar.symbol and item.close_time <= bar.close_time
        )
        context = builder.build(
            symbol=bar.symbol,
            as_of=bar.close_time,
            primary_timeframe=primary,
            bars=available,
        )
        frames.append(
            PaperFrame(
                context=context,
                quote=Quote(
                    symbol=bar.symbol,
                    bid=bar.bid.close,
                    ask=bar.ask.close,
                    timestamp=bar.close_time,
                ),
            )
        )
    return tuple(frames)


@dataclass(frozen=True, slots=True)
class PaperReplaySummary:
    cycles: int
    accepted_orders: int
    rejected_orders: int
    fills: int
    decision_counts: tuple[tuple[str, int], ...]
    ending_balance: float
    ending_equity: float
    realized_pnl: float
    unrealized_pnl: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "fills": self.fills,
            "decision_counts": dict(self.decision_counts),
            "ending_balance": self.ending_balance,
            "ending_equity": self.ending_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
        }


def run_replay(
    runtime: PaperIntegrationRuntime,
    frames: tuple[PaperFrame, ...],
) -> tuple[PaperReplaySummary, tuple[PaperCycleResult, ...]]:
    results = tuple(runtime.process(frame) for frame in frames)
    counts = Counter(result.decision.action.value for result in results)
    account = runtime.ledger.view()
    summary = PaperReplaySummary(
        cycles=len(results),
        accepted_orders=sum(result.accepted_orders for result in results),
        rejected_orders=sum(result.rejected_orders for result in results),
        fills=sum(result.sync.new_fills for result in results),
        decision_counts=tuple(sorted(counts.items())),
        ending_balance=account.balance,
        ending_equity=account.equity,
        realized_pnl=account.realized_pnl,
        unrealized_pnl=account.unrealized_pnl,
    )
    return summary, results
