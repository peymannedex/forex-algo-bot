"""Live and paper execution contracts, routing, and reconciliation."""

from fxbot.execution.broker import (
    BrokerAdapter,
    ExecutionError,
    FillSink,
    OrderNotFoundError,
    OrderSink,
    PermanentBrokerError,
    RiskAuthorizer,
    TransientBrokerError,
    UnknownSubmissionError,
    ensure_unique_fills,
)
from fxbot.execution.idempotency import (
    IdempotencyConflictError,
    IdempotencyRecord,
    InMemoryIdempotencyStore,
)
from fxbot.execution.lifecycle import (
    InvalidOrderTransitionError,
    transition_allowed,
    validate_transition,
    with_status,
)
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionFill,
    ExecutionMode,
    Metadata,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    RiskDecision,
    TimeInForce,
)
from fxbot.execution.paper import PaperBroker, PaperBrokerConfig
from fxbot.execution.router import (
    AllowAllRiskAuthorizer,
    ExecutionRouter,
    RetryPolicy,
)
from fxbot.execution.runtime import ExecutionRuntime, SyncResult
from fxbot.execution.safety import ExecutionControl, ExecutionControlState

__all__ = [
    "AllowAllRiskAuthorizer",
    "BrokerAdapter",
    "BrokerOrder",
    "ExecutionControl",
    "ExecutionControlState",
    "ExecutionError",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionFill",
    "ExecutionMode",
    "ExecutionRouter",
    "ExecutionRuntime",
    "FillSink",
    "IdempotencyConflictError",
    "IdempotencyRecord",
    "InMemoryIdempotencyStore",
    "InvalidOrderTransitionError",
    "Metadata",
    "OrderIntent",
    "OrderNotFoundError",
    "OrderSide",
    "OrderSink",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "PaperBrokerConfig",
    "PermanentBrokerError",
    "Quote",
    "RetryPolicy",
    "RiskAuthorizer",
    "RiskDecision",
    "SyncResult",
    "TimeInForce",
    "TransientBrokerError",
    "UnknownSubmissionError",
    "ensure_unique_fills",
    "transition_allowed",
    "validate_transition",
    "with_status",
]
