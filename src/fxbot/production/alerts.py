"""Operational alerts and composable alert sinks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class OperationalAlert:
    """One operator-facing production alert."""

    severity: AlertSeverity
    code: str
    message: str
    timestamp: datetime
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", AlertSeverity(self.severity))
        code = self.code.strip().upper()
        message = self.message.strip()
        if not code or not message:
            raise ValueError("alert code and message cannot be empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata)))


class AlertSink(Protocol):
    def emit(self, alert: OperationalAlert) -> None: ...


class InMemoryAlertSink:
    """Thread-safe sink used by tests and embedded status endpoints."""

    def __init__(self) -> None:
        self._alerts: list[OperationalAlert] = []
        self._lock = RLock()

    @property
    def alerts(self) -> tuple[OperationalAlert, ...]:
        with self._lock:
            return tuple(self._alerts)

    def emit(self, alert: OperationalAlert) -> None:
        with self._lock:
            self._alerts.append(alert)


class FanoutAlertSink:
    """Publish each alert to every configured sink."""

    def __init__(self, sinks: tuple[AlertSink, ...]) -> None:
        self.sinks = sinks

    def emit(self, alert: OperationalAlert) -> None:
        for sink in self.sinks:
            sink.emit(alert)


class LoggingAlertSink:
    """Write alerts to the standard logging pipeline."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("fxbot.alerts")

    def emit(self, alert: OperationalAlert) -> None:
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }[alert.severity]
        self.logger.log(
            level,
            alert.message,
            extra={
                "alert_code": alert.code,
                "alert_severity": alert.severity.value,
                "alert_metadata": dict(alert.metadata),
            },
        )


def alert(
    severity: AlertSeverity,
    code: str,
    message: str,
    *,
    timestamp: datetime | None = None,
    metadata: tuple[tuple[str, str], ...] = (),
) -> OperationalAlert:
    """Create an alert using the current UTC time by default."""

    return OperationalAlert(
        severity=severity,
        code=code,
        message=message,
        timestamp=timestamp or datetime.now(UTC),
        metadata=metadata,
    )
