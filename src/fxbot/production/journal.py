"""Durable execution journal for restart-safe downstream fill processing."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import cast

from fxbot.execution.broker import FillSink
from fxbot.execution.models import ExecutionFill

Clock = Callable[[], datetime]


class JournalState(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"


@dataclass(frozen=True, slots=True)
class ExecutionJournalEntry:
    execution_id: str
    state: JournalState
    updated_at: datetime

    def __post_init__(self) -> None:
        identifier = self.execution_id.strip()
        if not identifier:
            raise ValueError("execution_id cannot be empty")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        object.__setattr__(self, "execution_id", identifier)
        object.__setattr__(self, "state", JournalState(self.state))
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(UTC))


class FileExecutionJournal:
    """Atomically persist pending and committed execution IDs."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._entries = self._load()

    def state(self, execution_id: str) -> JournalState | None:
        with self._lock:
            entry = self._entries.get(execution_id)
            return entry.state if entry is not None else None

    @property
    def pending_execution_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(
                    execution_id
                    for execution_id, entry in self._entries.items()
                    if entry.state is JournalState.PENDING
                )
            )

    def begin(self, execution_id: str) -> bool:
        """Mark an execution pending; return False when already committed."""

        identifier = execution_id.strip()
        if not identifier:
            raise ValueError("execution_id cannot be empty")
        with self._lock:
            current = self._entries.get(identifier)
            if current is not None and current.state is JournalState.COMMITTED:
                return False
            self._entries[identifier] = ExecutionJournalEntry(
                identifier,
                JournalState.PENDING,
                self._clock(),
            )
            self._save()
            return True

    def commit(self, execution_id: str) -> None:
        identifier = execution_id.strip()
        if not identifier:
            raise ValueError("execution_id cannot be empty")
        with self._lock:
            self._entries[identifier] = ExecutionJournalEntry(
                identifier,
                JournalState.COMMITTED,
                self._clock(),
            )
            self._save()

    def _load(self) -> dict[str, ExecutionJournalEntry]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("execution journal root must be an object")
        payload = cast(dict[str, object], raw)
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("execution journal entries must be a list")
        output: dict[str, ExecutionJournalEntry] = {}
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError("execution journal entry must be an object")
            item = cast(dict[str, object], raw_entry)
            entry = ExecutionJournalEntry(
                execution_id=str(item["execution_id"]),
                state=JournalState(str(item["state"])),
                updated_at=datetime.fromisoformat(str(item["updated_at"])),
            )
            output[entry.execution_id] = entry
        return output

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "version": 1,
            "entries": [
                {
                    "execution_id": entry.execution_id,
                    "state": entry.state.value,
                    "updated_at": entry.updated_at.isoformat(),
                }
                for entry in sorted(
                    self._entries.values(),
                    key=lambda item: item.execution_id,
                )
            ],
        }
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class RecoverableFillSink:
    """Retry pending fills after restart and suppress committed executions."""

    def __init__(
        self,
        journal: FileExecutionJournal,
        downstream: FillSink,
    ) -> None:
        self.journal = journal
        self.downstream = downstream

    def on_fill(self, fill: ExecutionFill) -> None:
        if not self.journal.begin(fill.execution_id):
            return
        self.downstream.on_fill(fill)
        self.journal.commit(fill.execution_id)
