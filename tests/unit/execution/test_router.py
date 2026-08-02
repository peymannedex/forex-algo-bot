from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxbot.execution.broker import (
    PermanentBrokerError,
    TransientBrokerError,
    UnknownSubmissionError,
)
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionEventKind,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    RiskDecision,
)
from fxbot.execution.paper import PaperBroker
from fxbot.execution.router import ExecutionRouter, RetryPolicy

BASE = datetime(2026, 1, 5, tzinfo=UTC)


class Risk:
    def __init__(self, decision: RiskDecision) -> None:
        self.decision = decision
        self.calls = 0

    def authorize(self, intent: OrderIntent) -> RiskDecision:
        self.calls += 1
        return self.decision


class FlakyBroker(PaperBroker):
    def __init__(self, failures: int, *, unknown: bool = False) -> None:
        super().__init__()
        self.failures = failures
        self.unknown = unknown
        self.submit_calls = 0

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        self.submit_calls += 1
        if self.submit_calls <= self.failures:
            if self.unknown:
                raise UnknownSubmissionError("timeout")
            raise TransientBrokerError("temporary")
        return super().submit_order(intent)


class UnknownButAcceptedBroker(PaperBroker):
    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        if self.first:
            self.first = False
            super().submit_order(intent)
            raise UnknownSubmissionError("connection lost after acceptance")
        return super().submit_order(intent)


def market_intent(quantity: float = 1.0) -> OrderIntent:
    return OrderIntent(
        "client",
        "key",
        "EURUSD",
        OrderSide.BUY,
        OrderType.MARKET,
        quantity,
        BASE,
    )


def router_for(broker: PaperBroker, **kwargs: object) -> ExecutionRouter:
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    return ExecutionRouter(
        broker,
        clock=lambda: BASE,
        sleeper=lambda _: None,
        **kwargs,
    )


def test_submit_routes_and_audits() -> None:
    broker = PaperBroker()
    router = router_for(broker)
    order = router.submit(market_intent())
    assert order.status is OrderStatus.FILLED
    assert [event.kind for event in router.audit_events] == [
        ExecutionEventKind.RISK_APPROVED,
        ExecutionEventKind.ORDER_SUBMITTED,
        ExecutionEventKind.ORDER_ACKNOWLEDGED,
    ]


def test_duplicate_submission_returns_prior_order() -> None:
    broker = PaperBroker()
    router = router_for(broker)
    first = router.submit(market_intent())
    second = router.submit(market_intent())
    assert first == second
    assert len(broker.orders) == 1
    assert router.audit_events[-1].kind is ExecutionEventKind.DUPLICATE_SUPPRESSED


def test_risk_rejection_prevents_broker_submission() -> None:
    broker = PaperBroker()
    risk = Risk(RiskDecision(False, "daily loss limit"))
    router = router_for(broker, risk_authorizer=risk)
    with pytest.raises(PermanentBrokerError, match="daily loss limit"):
        router.submit(market_intent())
    assert broker.orders == ()
    assert router.audit_events[-1].kind is ExecutionEventKind.RISK_REJECTED


def test_risk_can_reduce_quantity() -> None:
    broker = PaperBroker()
    risk = Risk(RiskDecision(True, "reduced", 0.25))
    router = router_for(broker, risk_authorizer=risk)
    order = router.submit(market_intent(1.0))
    assert order.requested_quantity == pytest.approx(0.25)


def test_risk_cannot_increase_quantity() -> None:
    broker = PaperBroker()
    risk = Risk(RiskDecision(True, "bad", 2.0))
    router = router_for(broker, risk_authorizer=risk)
    with pytest.raises(ValueError, match="cannot increase"):
        router.submit(market_intent(1.0))


def test_transient_submission_is_retried() -> None:
    broker = FlakyBroker(2)
    policy = RetryPolicy(max_attempts=3, initial_delay_seconds=0)
    router = router_for(broker, retry_policy=policy)
    order = router.submit(market_intent())
    assert order.status is OrderStatus.FILLED
    assert broker.submit_calls == 3
    assert sum(e.kind is ExecutionEventKind.RETRY for e in router.audit_events) == 2


def test_unknown_submission_is_resolved_by_client_id() -> None:
    broker = UnknownButAcceptedBroker()
    router = router_for(broker)
    order = router.submit(market_intent())
    assert order.status is OrderStatus.FILLED
    assert len(broker.orders) == 1


def test_retry_exhaustion_reraises() -> None:
    broker = FlakyBroker(3)
    router = router_for(broker, retry_policy=RetryPolicy(max_attempts=2))
    with pytest.raises(TransientBrokerError):
        router.submit(market_intent())


def test_cancel_active_order() -> None:
    broker = PaperBroker()
    broker.update_quote(Quote("EURUSD", 1.1, 1.1002, BASE))
    pending = broker.submit_order(
        OrderIntent(
            "pending",
            "pending-key",
            "EURUSD",
            OrderSide.BUY,
            OrderType.LIMIT,
            1,
            BASE,
            limit_price=1.0,
        )
    )
    router = ExecutionRouter(broker, clock=lambda: BASE, sleeper=lambda _: None)
    cancelled = router.cancel(pending.broker_order_id)
    assert cancelled.status is OrderStatus.CANCELLED
    assert router.audit_events[-1].kind is ExecutionEventKind.ORDER_CANCELLED


def test_cancel_terminal_order_is_noop() -> None:
    broker = PaperBroker()
    router = router_for(broker)
    filled = router.submit(market_intent())
    before = len(router.audit_events)
    assert router.cancel(filled.broker_order_id) == filled
    assert len(router.audit_events) == before


def test_retry_policy_delays() -> None:
    policy = RetryPolicy(4, 0.5, 2, 1.5)
    assert [policy.delay_before_attempt(i) for i in range(1, 5)] == [0, 0.5, 1.0, 1.5]
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        policy.delay_before_attempt(0)


def test_kill_switch_blocks_submission() -> None:
    from fxbot.execution.safety import ExecutionControl

    broker = PaperBroker()
    control = ExecutionControl.disarmed("operator stop", clock=lambda: BASE)
    router = router_for(broker, control=control)
    with pytest.raises(PermanentBrokerError, match="operator stop"):
        router.submit(market_intent())
    assert router.audit_events[-1].kind is ExecutionEventKind.SAFETY_BLOCKED
