from datetime import UTC, datetime

from fxbot.production.checkpoint import (
    SupervisorCheckpoint,
    SupervisorCheckpointStore,
)


def test_missing_checkpoint_returns_empty(tmp_path) -> None:
    store = SupervisorCheckpointStore(tmp_path / "state.json")

    assert store.load() == SupervisorCheckpoint()


def test_round_trip(tmp_path) -> None:
    store = SupervisorCheckpointStore(tmp_path / "state.json")
    checkpoint = SupervisorCheckpoint(
        last_heartbeat_at=datetime(2026, 1, 1, tzinfo=UTC),
        last_reconciliation_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        last_seen_execution_id="fill-1",
    )

    store.save(checkpoint)

    assert store.load() == checkpoint
    assert not (tmp_path / "state.json.tmp").exists()
