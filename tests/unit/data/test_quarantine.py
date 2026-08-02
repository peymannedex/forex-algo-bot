from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from fxbot.data.adapters.base import AdapterDiagnostics
from fxbot.data.cleaning import CleaningReport, DataQualityIssue
from fxbot.data.quality import DataQualityGate, QualityThresholds
from fxbot.data.quarantine import JsonQuarantineStore, QuarantineEntry
from fxbot.domain.enums import DataKind, Timeframe

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def entry() -> QuarantineEntry:
    report = CleaningReport(
        input_records=10,
        output_records=8,
        spread_rejections=2,
        issues=(
            DataQualityIssue(
                "spread_limit",
                "spread too wide",
                symbol="EURUSD",
                timestamp=BASE,
            ),
        ),
    )
    decision = DataQualityGate(
        QualityThresholds(max_rejection_rate=0.10)
    ).evaluate(report, AdapterDiagnostics(rows_read=10, records_emitted=10))
    return QuarantineEntry(
        pipeline_id="historical",
        source="mt5",
        symbol="eurusd",
        kind=DataKind.TICK,
        timeframe=Timeframe.TICK,
        start_time=BASE,
        end_time=BASE + timedelta(hours=1),
        fetched_records=10,
        diagnostics=AdapterDiagnostics(rows_read=10, records_emitted=10),
        cleaning_report=report,
        decision=decision,
        sample_records=({"kind": "tick", "bid": "0x1.0p+0"},),
        created_at=BASE + timedelta(days=1),
    )


def test_json_quarantine_store_writes_partitioned_atomic_audit(tmp_path) -> None:
    store = JsonQuarantineStore(tmp_path / "quarantine")

    path = store.write(entry())
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent.relative_to(tmp_path).as_posix() == (
        "quarantine/kind=tick/symbol=EURUSD/year=2026/month=01"
    )
    assert payload["quarantine_schema"] == 1
    assert payload["range"]["semantics"] == "half-open"
    assert payload["quality"]["accepted"] is False
    assert payload["cleaning_report"]["issues"][0]["timestamp"].startswith("2026-01-01")
    assert not list(path.parent.glob("*.tmp"))


def test_quarantine_entry_enforces_timeframe_and_metadata_rules() -> None:
    valid = entry()
    assert valid.symbol == "EURUSD"
    assert valid.timeframe is None

    with pytest.raises(ValueError, match="timezone-aware"):
        QuarantineEntry(
            **{
                **{
                    field: getattr(valid, field)
                    for field in valid.__dataclass_fields__
                    if field not in {"start_time", "timeframe"}
                },
                "start_time": datetime(2026, 1, 1),
                "timeframe": None,
            }
        )
    with pytest.raises(ValueError, match="Bar quarantine"):
        QuarantineEntry(
            pipeline_id="historical",
            source="mt5",
            symbol="EURUSD",
            kind=DataKind.BAR,
            timeframe=None,
            start_time=BASE,
            end_time=BASE + timedelta(minutes=1),
            fetched_records=1,
            diagnostics=valid.diagnostics,
            cleaning_report=valid.cleaning_report,
            decision=valid.decision,
        )
