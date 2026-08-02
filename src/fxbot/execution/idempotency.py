"""Thread-safe idempotency registry for at-most-once order submission."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock

from fxbot.execution.models import BrokerOrder, OrderIntent


class IdempotencyConflictError(ValueError):
    """Raised when one key is reused for a semantically different order."""


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """Durable mapping between an idempotency key and broker outcome."""

    key: str
    semantic_fingerprint: str
    client_order_id: str
    created_at: datetime
    broker_order: BrokerOrder | None = None


class InMemoryIdempotencyStore:
    """Small process-local idempotency store suitable for tests and paper mode."""

    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = RLock()

    def reserve(self, intent: OrderIntent) -> IdempotencyRecord:
        """Reserve a key or return its existing compatible record."""

        with self._lock:
            existing = self._records.get(intent.idempotency_key)
            if existing is not None:
                if existing.semantic_fingerprint != intent.semantic_fingerprint:
                    raise IdempotencyConflictError(
                        f"Idempotency key reused for different order: {intent.idempotency_key}"
                    )
                return existing
            record = IdempotencyRecord(
                key=intent.idempotency_key,
                semantic_fingerprint=intent.semantic_fingerprint,
                client_order_id=intent.client_order_id,
                created_at=datetime.now(UTC),
            )
            self._records[intent.idempotency_key] = record
            return record

    def bind(self, intent: OrderIntent, order: BrokerOrder) -> IdempotencyRecord:
        """Attach the broker order to a previously reserved compatible key."""

        with self._lock:
            record = self.reserve(intent)
            if order.client_order_id != record.client_order_id:
                raise IdempotencyConflictError("Broker order client_order_id does not match")
            if record.broker_order is not None:
                if record.broker_order.broker_order_id != order.broker_order_id:
                    raise IdempotencyConflictError("Idempotency key is already bound elsewhere")
                return record
            bound = IdempotencyRecord(
                key=record.key,
                semantic_fingerprint=record.semantic_fingerprint,
                client_order_id=record.client_order_id,
                created_at=record.created_at,
                broker_order=order,
            )
            self._records[intent.idempotency_key] = bound
            return bound

    def get(self, key: str) -> IdempotencyRecord | None:
        """Return a record without mutating it."""

        with self._lock:
            return self._records.get(key)

    def clear(self) -> None:
        """Remove all records; intended only for tests and controlled resets."""

        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
