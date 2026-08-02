"""Idempotent order routing with pre-trade risk and bounded retry safety."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from time import sleep

from fxbot.execution.broker import (
    BrokerAdapter,
    PermanentBrokerError,
    RiskAuthorizer,
    TransientBrokerError,
    UnknownSubmissionError,
)
from fxbot.execution.idempotency import InMemoryIdempotencyStore
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionEvent,
    ExecutionEventKind,
    OrderIntent,
    RiskDecision,
)
from fxbot.execution.safety import ExecutionControl

Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic bounded retry policy for transient broker failures."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.25
    multiplier: float = 2.0
    maximum_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        for name in ("initial_delay_seconds", "multiplier", "maximum_delay_seconds"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be at least one")

    def delay_before_attempt(self, attempt: int) -> float:
        """Return delay before a 1-based attempt; the first attempt is immediate."""

        if attempt < 1:
            raise ValueError("attempt must be at least one")
        if attempt == 1:
            return 0.0
        raw = self.initial_delay_seconds * self.multiplier ** (attempt - 2)
        return min(raw, self.maximum_delay_seconds)


class AllowAllRiskAuthorizer:
    """Default pre-trade authorizer used when no Phase 2 adapter is supplied."""

    def authorize(self, intent: OrderIntent) -> RiskDecision:
        return RiskDecision(True, "approved", approved_quantity=intent.quantity)


class ExecutionRouter:
    """Route normalized orders with risk checks and at-most-once semantics."""

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        risk_authorizer: RiskAuthorizer | None = None,
        idempotency_store: InMemoryIdempotencyStore | None = None,
        retry_policy: RetryPolicy | None = None,
        clock: Clock | None = None,
        sleeper: Sleeper = sleep,
        control: ExecutionControl | None = None,
    ) -> None:
        self.broker = broker
        self.risk_authorizer: RiskAuthorizer = (
            risk_authorizer if risk_authorizer is not None else AllowAllRiskAuthorizer()
        )
        self.idempotency_store = (
            idempotency_store if idempotency_store is not None else InMemoryIdempotencyStore()
        )
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.control = control if control is not None else ExecutionControl.armed()
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._sleeper: Sleeper = sleeper
        self._events: list[ExecutionEvent] = []
        self._sequence = 0

    @property
    def audit_events(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._events)

    def submit(self, intent: OrderIntent) -> BrokerOrder:
        """Authorize and submit an order exactly once per idempotency key."""

        try:
            self.control.ensure_enabled()
        except PermanentBrokerError as exc:
            self._audit(
                ExecutionEventKind.SAFETY_BLOCKED,
                str(exc),
                intent=intent,
            )
            raise

        record = self.idempotency_store.reserve(intent)
        if record.broker_order is not None:
            self._audit(
                ExecutionEventKind.DUPLICATE_SUPPRESSED,
                "Duplicate submission returned prior broker order",
                intent=intent,
                order=record.broker_order,
            )
            return record.broker_order

        decision = self.risk_authorizer.authorize(intent)
        if not decision.approved:
            self._audit(
                ExecutionEventKind.RISK_REJECTED,
                decision.reason,
                intent=intent,
            )
            raise PermanentBrokerError(f"Pre-trade risk rejected order: {decision.reason}")

        approved_intent = self._apply_approved_quantity(intent, decision)
        self._audit(
            ExecutionEventKind.RISK_APPROVED,
            decision.reason,
            intent=approved_intent,
        )

        order = self._submit_with_retry(approved_intent)
        self.idempotency_store.bind(intent, order)
        self._audit(
            ExecutionEventKind.ORDER_ACKNOWLEDGED,
            "Broker order accepted",
            intent=approved_intent,
            order=order,
        )
        return order

    def cancel(self, broker_order_id: str) -> BrokerOrder:
        """Cancel an active order with bounded retries."""

        order = self.broker.get_order(broker_order_id)
        if order.status.terminal:
            return order
        self._audit(
            ExecutionEventKind.ORDER_CANCEL_REQUESTED,
            "Cancellation requested",
            order=order,
        )
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            self._delay(attempt)
            try:
                cancelled = self.broker.cancel_order(broker_order_id)
            except TransientBrokerError:
                if attempt >= self.retry_policy.max_attempts:
                    raise
                self._audit(
                    ExecutionEventKind.RETRY,
                    f"Retrying cancellation after transient failure ({attempt})",
                    order=order,
                    metadata=(("attempt", str(attempt)),),
                )
                continue
            kind = (
                ExecutionEventKind.ORDER_CANCELLED
                if cancelled.status.terminal
                else ExecutionEventKind.ORDER_CANCEL_REQUESTED
            )
            self._audit(kind, "Cancellation state updated", order=cancelled)
            return cancelled
        raise AssertionError("unreachable")

    def _submit_with_retry(self, intent: OrderIntent) -> BrokerOrder:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            self._delay(attempt)
            self._audit(
                ExecutionEventKind.ORDER_SUBMITTED,
                "Submitting order to broker",
                intent=intent,
                metadata=(("attempt", str(attempt)),),
            )
            try:
                return self.broker.submit_order(intent)
            except UnknownSubmissionError:
                resolved = self.broker.find_order_by_client_id(intent.client_order_id)
                if resolved is not None:
                    return resolved
                if attempt >= self.retry_policy.max_attempts:
                    raise
                self._audit(
                    ExecutionEventKind.RETRY,
                    "Unknown submission not found by client ID; retrying",
                    intent=intent,
                    metadata=(("attempt", str(attempt)),),
                )
            except TransientBrokerError:
                if attempt >= self.retry_policy.max_attempts:
                    raise
                self._audit(
                    ExecutionEventKind.RETRY,
                    "Retrying order after transient broker failure",
                    intent=intent,
                    metadata=(("attempt", str(attempt)),),
                )
        raise AssertionError("unreachable")

    @staticmethod
    def _apply_approved_quantity(intent: OrderIntent, decision: RiskDecision) -> OrderIntent:
        quantity = decision.approved_quantity
        if quantity is None or abs(quantity - intent.quantity) <= 1e-12:
            return intent
        if quantity > intent.quantity + 1e-12:
            raise ValueError("Risk authorizer cannot increase requested quantity")
        return replace(intent, quantity=quantity)

    def _delay(self, attempt: int) -> None:
        delay = self.retry_policy.delay_before_attempt(attempt)
        if delay > 0.0:
            self._sleeper(delay)

    def _audit(
        self,
        kind: ExecutionEventKind,
        message: str,
        *,
        intent: OrderIntent | None = None,
        order: BrokerOrder | None = None,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._events.append(
            ExecutionEvent(
                sequence=self._sequence,
                timestamp=self._clock(),
                kind=kind,
                message=message,
                client_order_id=(
                    intent.client_order_id
                    if intent is not None
                    else order.client_order_id if order is not None else None
                ),
                broker_order_id=(order.broker_order_id if order is not None else None),
                metadata=metadata,
            )
        )
        self._sequence += 1
