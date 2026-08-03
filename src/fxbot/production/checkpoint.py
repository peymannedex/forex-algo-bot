"""Atomic supervisor checkpoint persistence for restart recovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SupervisorCheckpoint:
    last_heartbeat_at: datetime | None = None
    last_reconciliation_at: datetime | None = None
    last_seen_execution_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("last_heartbeat_at", "last_reconciliation_at"):
            value = getattr(self, name)
            if value is not None:
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{name} must be timezone-aware")
                object.__setattr__(self, name, value.astimezone(UTC))

    def to_dict(self) -> dict[str, str | None]:
        return {
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat()
                if self.last_heartbeat_at is not None
                else None
            ),
            "last_reconciliation_at": (
                self.last_reconciliation_at.isoformat()
                if self.last_reconciliation_at is not None
                else None
            ),
            "last_seen_execution_id": self.last_seen_execution_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SupervisorCheckpoint:
        def parse(name: str) -> datetime | None:
            value = data.get(name)
            return datetime.fromisoformat(str(value)) if value else None

        execution_id = data.get("last_seen_execution_id")
        return cls(
            last_heartbeat_at=parse("last_heartbeat_at"),
            last_reconciliation_at=parse("last_reconciliation_at"),
            last_seen_execution_id=str(execution_id) if execution_id else None,
        )


class SupervisorCheckpointStore:
    """Write checkpoints atomically to avoid partial files after a crash."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> SupervisorCheckpoint:
        if not self.path.exists():
            return SupervisorCheckpoint()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be an object")
        return SupervisorCheckpoint.from_dict(data)

    def save(self, checkpoint: SupervisorCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(checkpoint.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
