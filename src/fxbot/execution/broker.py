"""Broker adapter protocol and execution-layer error taxonomy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from fxbot.execution.models import BrokerOrder, ExecutionFill, OrderIntent, Quote, RiskDecision


class ExecutionError(RuntimeError):
    """Base error raised by the execution layer."""


class TransientBrokerError(ExecutionError):
    """Retryable connectivity, timeout, or broker-availability error."""


class PermanentBrokerError(ExecutionError):
    """Non-retryable broker validation or authorization error."""


class UnknownSubmissionError(TransientBrokerError):
    """Submission outcome is unknown and must be resolved by client-order lookup."""


class OrderNotFoundError(PermanentBrokerError):
    """Requested broker order does not exist."""


@runtime_checkable
class BrokerAdapter(Protocol):
    """Minimal synchronous contract implemented by paper and live brokers."""

    @property
    def name(self) -> str:
        """Return a stable broker adapter name."""
        ...

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        """Submit one normalized order intent."""
        ...

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        """Request cancellation and return the latest broker state."""
        ...

    def get_order(self, broker_order_id: str) -> BrokerOrder:
        """Return the latest state for one broker order."""
        ...

    def find_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Resolve a potentially unknown submission by client order ID."""
        ...

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        """Return currently active orders."""
        ...

    def drain_fills(self) -> tuple[ExecutionFill, ...]:
        """Return and clear fills not yet consumed by the runtime."""
        ...

    def update_quote(self, quote: Quote) -> tuple[ExecutionFill, ...]:
        """Process a new quote and return fills produced by it."""
        ...


class FillSink(Protocol):
    """Observer receiving unique fills after broker polling."""

    def on_fill(self, fill: ExecutionFill) -> None:
        """Process one previously unseen fill."""
        ...


class OrderSink(Protocol):
    """Observer receiving durable order state changes."""

    def on_order(self, order: BrokerOrder) -> None:
        """Process one order state update."""
        ...


class RiskAuthorizer(Protocol):
    """Pre-trade risk seam implemented by Phase 2 adapters."""

    def authorize(self, intent: OrderIntent) -> RiskDecision:
        """Return the pre-trade authorization decision."""
        ...


def ensure_unique_fills(fills: Iterable[ExecutionFill]) -> tuple[ExecutionFill, ...]:
    """Return fills in input order while rejecting duplicate execution IDs."""

    output: list[ExecutionFill] = []
    seen: set[str] = set()
    for fill in fills:
        if fill.execution_id in seen:
            raise ExecutionError(f"Duplicate execution_id: {fill.execution_id}")
        seen.add(fill.execution_id)
        output.append(fill)
    return tuple(output)
