from datetime import UTC, datetime

import pytest

from fxbot.execution.models import ExecutionFill, OrderSide
from fxbot.production.journal import (
    FileExecutionJournal,
    JournalState,
    RecoverableFillSink,
)


class Sink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.fills = []

    def on_fill(self, fill) -> None:
        self.fills.append(fill)
        if self.fail:
            raise RuntimeError("downstream failed")


def fill(identifier: str = "fill-1") -> ExecutionFill:
    return ExecutionFill(
        execution_id=identifier,
        broker_order_id="order-1",
        client_order_id="client-1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=0.1,
        price=1.1,
        executed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_committed_fill_is_suppressed_after_restart(tmp_path) -> None:
    path = tmp_path / "journal.json"
    sink = Sink()
    wrapper = RecoverableFillSink(FileExecutionJournal(path), sink)

    wrapper.on_fill(fill())
    RecoverableFillSink(FileExecutionJournal(path), sink).on_fill(fill())

    assert len(sink.fills) == 1
    assert FileExecutionJournal(path).state("fill-1") is JournalState.COMMITTED


def test_failed_fill_remains_pending_and_retries(tmp_path) -> None:
    path = tmp_path / "journal.json"
    failing = Sink(fail=True)
    wrapper = RecoverableFillSink(FileExecutionJournal(path), failing)

    with pytest.raises(RuntimeError):
        wrapper.on_fill(fill())

    journal = FileExecutionJournal(path)
    assert journal.state("fill-1") is JournalState.PENDING
    assert journal.pending_execution_ids == ("fill-1",)

    healthy = Sink()
    RecoverableFillSink(journal, healthy).on_fill(fill())

    assert len(healthy.fills) == 1
    assert journal.state("fill-1") is JournalState.COMMITTED
