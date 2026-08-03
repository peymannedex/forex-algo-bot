"""Health, readiness, and liveness aggregation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock

Clock = Callable[[], datetime]


class ComponentState(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    state: ComponentState
    message: str
    checked_at: datetime
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("component name cannot be empty")
        if self.checked_at.tzinfo is None or self.checked_at.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "state", ComponentState(self.state))
        object.__setattr__(self, "message", self.message.strip() or self.state.value)
        object.__setattr__(self, "checked_at", self.checked_at.astimezone(UTC))
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata)))


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    generated_at: datetime
    components: tuple[ComponentHealth, ...]

    @property
    def live(self) -> bool:
        return any(component.state is not ComponentState.STOPPED for component in self.components)

    @property
    def ready(self) -> bool:
        return bool(self.components) and all(
            component.state is ComponentState.HEALTHY for component in self.components
        )

    @property
    def state(self) -> ComponentState:
        states = {component.state for component in self.components}
        if ComponentState.UNHEALTHY in states:
            return ComponentState.UNHEALTHY
        if ComponentState.DEGRADED in states or ComponentState.UNKNOWN in states:
            return ComponentState.DEGRADED
        if states == {ComponentState.STOPPED}:
            return ComponentState.STOPPED
        return ComponentState.HEALTHY if self.components else ComponentState.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "state": self.state.value,
            "live": self.live,
            "ready": self.ready,
            "components": [
                {
                    "name": component.name,
                    "state": component.state.value,
                    "message": component.message,
                    "checked_at": component.checked_at.isoformat(),
                    "metadata": dict(component.metadata),
                }
                for component in self.components
            ],
        }


class HealthRegistry:
    """Thread-safe registry with automatic stale-component degradation."""

    def __init__(
        self,
        *,
        stale_after: timedelta | None = None,
        clock: Clock | None = None,
    ) -> None:
        resolved_stale_after = stale_after or timedelta(seconds=20)
        if resolved_stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self.stale_after = resolved_stale_after
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._components: dict[str, ComponentHealth] = {}
        self._lock = RLock()

    def update(
        self,
        name: str,
        state: ComponentState,
        message: str,
        *,
        metadata: tuple[tuple[str, str], ...] = (),
        checked_at: datetime | None = None,
    ) -> ComponentHealth:
        component = ComponentHealth(
            name=name,
            state=state,
            message=message,
            checked_at=checked_at or self._clock(),
            metadata=metadata,
        )
        with self._lock:
            self._components[component.name] = component
        return component

    def stop_all(self, message: str = "service stopped") -> None:
        now = self._clock()
        with self._lock:
            names = tuple(self._components)
            for name in names:
                self._components[name] = ComponentHealth(
                    name=name,
                    state=ComponentState.STOPPED,
                    message=message,
                    checked_at=now,
                )

    def snapshot(self) -> HealthSnapshot:
        now = self._clock()
        with self._lock:
            values = []
            for component in self._components.values():
                age = now - component.checked_at
                if (
                    age > self.stale_after
                    and component.state not in {ComponentState.UNHEALTHY, ComponentState.STOPPED}
                ):
                    component = ComponentHealth(
                        name=component.name,
                        state=ComponentState.DEGRADED,
                        message=f"stale health check ({age.total_seconds():.1f}s old)",
                        checked_at=component.checked_at,
                        metadata=component.metadata,
                    )
                values.append(component)
        values.sort(key=lambda item: item.name)
        return HealthSnapshot(generated_at=now, components=tuple(values))
