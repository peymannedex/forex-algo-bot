"""Atomic JSON quarantine records for rejected market-data batches."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fxbot.data.adapters.base import AdapterDiagnostics
from fxbot.data.cleaning import CleaningReport
from fxbot.data.quality import QualityDecision
from fxbot.domain.enums import DataKind, Timeframe


class QuarantineError(RuntimeError):
    """Raised when a rejected-batch audit record cannot be persisted."""


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    """Audit metadata for one rejected half-open ingestion chunk."""

    pipeline_id: str
    source: str
    symbol: str
    kind: DataKind
    timeframe: Timeframe | None
    start_time: datetime
    end_time: datetime
    fetched_records: int
    diagnostics: AdapterDiagnostics
    cleaning_report: CleaningReport
    decision: QualityDecision
    sample_records: tuple[dict[str, object], ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        pipeline_id = self.pipeline_id.strip()
        source = self.source.strip()
        symbol = self.symbol.strip().upper()
        kind = DataKind(self.kind)
        start = _utc(self.start_time, "start_time")
        end = _utc(self.end_time, "end_time")
        created = _utc(self.created_at, "created_at")
        if not pipeline_id:
            raise ValueError("pipeline_id cannot be empty")
        if not source:
            raise ValueError("source cannot be empty")
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if start >= end:
            raise ValueError("start_time must be earlier than end_time")
        if self.fetched_records < 0:
            raise ValueError("fetched_records must be non-negative")
        timeframe = self.timeframe
        if kind is DataKind.TICK:
            if timeframe is not None and Timeframe.parse(timeframe) is not Timeframe.TICK:
                raise ValueError("Tick quarantine entries cannot use a bar timeframe")
            timeframe = None
        else:
            if timeframe is None or Timeframe.parse(timeframe) is Timeframe.TICK:
                raise ValueError("Bar quarantine entries require a non-tick timeframe")
            timeframe = Timeframe.parse(timeframe)
        object.__setattr__(self, "pipeline_id", pipeline_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "created_at", created)


class JsonQuarantineStore:
    """Persist rejected-batch audits as immutable, atomically written JSON."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def write(self, entry: QuarantineEntry) -> Path:
        """Write one immutable quarantine file and return its final path."""

        directory = (
            self.root
            / f"kind={entry.kind.value}"
            / f"symbol={entry.symbol}"
            / f"year={entry.start_time.year:04d}"
            / f"month={entry.start_time.month:02d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        stamp = entry.created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        filename = f"rejected-{stamp}-{uuid4().hex}.json"
        destination = directory / filename
        temporary = directory / f".{filename}.tmp"
        payload = self._payload(entry)
        try:
            temporary.write_text(
                json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(temporary, destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise QuarantineError(f"Failed writing quarantine record {destination}: {exc}") from exc
        return destination

    @staticmethod
    def _payload(entry: QuarantineEntry) -> dict[str, object]:
        issues = [
            {
                "code": issue.code,
                "message": issue.message,
                "symbol": issue.symbol,
                "timestamp": (
                    issue.timestamp.astimezone(UTC).isoformat(timespec="microseconds")
                    if issue.timestamp is not None
                    else None
                ),
                "severity": issue.severity,
            }
            for issue in entry.cleaning_report.issues
        ]
        report = asdict(entry.cleaning_report)
        report["issues"] = issues
        return {
            "quarantine_schema": 1,
            "pipeline_id": entry.pipeline_id,
            "source": entry.source,
            "stream": {
                "kind": entry.kind.value,
                "symbol": entry.symbol,
                "timeframe": entry.timeframe.value if entry.timeframe is not None else None,
            },
            "range": {
                "start": entry.start_time.isoformat(timespec="microseconds"),
                "end": entry.end_time.isoformat(timespec="microseconds"),
                "semantics": "half-open",
            },
            "fetched_records": entry.fetched_records,
            "diagnostics": asdict(entry.diagnostics),
            "cleaning_report": report,
            "quality": {
                "accepted": entry.decision.accepted,
                "reasons": list(entry.decision.reasons),
                "metrics": asdict(entry.decision.metrics),
            },
            "sample_records": list(entry.sample_records),
            "created_at": entry.created_at.isoformat(timespec="microseconds"),
        }
