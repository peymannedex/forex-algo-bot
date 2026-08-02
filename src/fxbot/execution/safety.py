"""Operational execution control and emergency kill switch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock

from fxbot.execution.broker import PermanentBrokerError

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class ExecutionControlState:
    """Current operational state of the order-routing kill switch."""

    enabled: bool
    reason: str
    changed_at: datetime

    def __post_init__(self) -> None:
        if self.changed_at.tzinfo is None or self.changed_at.utcoffset() is None:
            raise ValueError("changed_at must be timezone-aware")
        object.__setattr__(self, "changed_at", self.changed_at.astimezone(UTC))
        object.__setattr__(self, "reason", self.reason.strip() or "unspecified")


class ExecutionControl:
    """Thread-safe manual and automated kill switch for order submission."""

    def __init__(
        self,
        *,
        enabled: bool,
        reason: str,
        clock: Clock | None = None,
    ) -> None:
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._state = ExecutionControlState(enabled, reason, self._clock())

    @classmethod
    def armed(cls, *, clock: Clock | None = None) -> ExecutionControl:
        return cls(enabled=True, reason="armed", clock=clock)

    @classmethod
    def disarmed(
        cls,
        reason: str = "not armed",
        *,
        clock: Clock | None = None,
    ) -> ExecutionControl:
        return cls(enabled=False, reason=reason, clock=clock)

    @property
    def state(self) -> ExecutionControlState:
        with self._lock:
            return self._state

    def arm(self, reason: str = "operator armed") -> ExecutionControlState:
        with self._lock:
            self._state = ExecutionControlState(True, reason, self._clock())
            return self._state

    def trip(self, reason: str) -> ExecutionControlState:
        with self._lock:
            self._state = ExecutionControlState(False, reason, self._clock())
            return self._state

    def ensure_enabled(self) -> None:
        state = self.state
        if not state.enabled:
            raise PermanentBrokerError(f"Execution is disabled: {state.reason}")
