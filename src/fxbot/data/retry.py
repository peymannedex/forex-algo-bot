"""Deterministic exponential-backoff policies for reconnecting data services."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """Bounded exponential backoff with optional symmetric jitter.

    ``attempt`` values are one-based: the first retry uses
    ``initial_delay_seconds``.  Jitter is calculated as a fraction of the
    bounded delay, where a random value of ``0`` applies the maximum negative
    jitter and ``1`` applies the maximum positive jitter.
    """

    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.10
    max_attempts: int | None = None

    def __post_init__(self) -> None:
        for name in ("initial_delay_seconds", "max_delay_seconds", "multiplier"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)
        jitter = float(self.jitter_ratio)
        if not isfinite(jitter) or not 0.0 <= jitter <= 1.0:
            raise ValueError("jitter_ratio must be between 0 and 1")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max_delay_seconds cannot be below initial_delay_seconds")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be at least 1")
        if self.max_attempts is not None and self.max_attempts < 0:
            raise ValueError("max_attempts must be non-negative or None")
        object.__setattr__(self, "jitter_ratio", jitter)

    def delay_seconds(self, attempt: int, *, random_value: float = 0.5) -> float:
        """Return the delay for a one-based retry attempt."""

        if attempt <= 0:
            raise ValueError("attempt must be one-based and positive")
        if not 0.0 <= random_value <= 1.0:
            raise ValueError("random_value must be between 0 and 1")
        base = min(
            self.max_delay_seconds,
            self.initial_delay_seconds * self.multiplier ** (attempt - 1),
        )
        if self.jitter_ratio == 0.0:
            return base
        signed = (random_value * 2.0) - 1.0
        return max(0.0, base * (1.0 + signed * self.jitter_ratio))

    def allows_retry(self, completed_attempts: int) -> bool:
        """Return whether another retry may start after prior attempts."""

        if completed_attempts < 0:
            raise ValueError("completed_attempts must be non-negative")
        return self.max_attempts is None or completed_attempts < self.max_attempts
