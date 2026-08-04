import json
from datetime import UTC, datetime, timedelta

from fxbot.integration.soak import SoakEvidenceWriter
from fxbot.production.health import ComponentState, HealthRegistry


def test_evidence_writer_records_errors_atomically(tmp_path) -> None:
    now = [datetime(2026, 1, 5, 14, 0, tzinfo=UTC)]
    health = HealthRegistry(clock=lambda: now[0])
    health.update("service", ComponentState.HEALTHY, "ready")
    writer = SoakEvidenceWriter(
        tmp_path,
        health=health,
        clock=lambda: now[0],
    )

    writer.record_error(RuntimeError("feed unavailable"))
    now[0] += timedelta(seconds=1)
    summary = writer.write_summary()

    assert summary.errors == 1
    payload = json.loads(
        (tmp_path / "latest-summary.json").read_text(encoding="utf-8")
    )
    assert payload["errors"] == 1
    assert (tmp_path / "errors.jsonl").exists()
    assert not (tmp_path / "latest-summary.json.tmp").exists()
