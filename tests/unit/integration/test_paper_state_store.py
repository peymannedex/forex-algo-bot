from datetime import UTC, datetime

from fxbot.integration.models import PaperPosition
from fxbot.integration.state import PaperRuntimeState, PaperRuntimeStateStore


def test_state_store_round_trip(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = PaperRuntimeStateStore(path)
    state = PaperRuntimeState(
        cycle=7,
        last_frame_at=datetime(2026, 1, 1, tzinfo=UTC),
        balance=100_100.0,
        day_start_equity=100_000.0,
        peak_equity=100_200.0,
        realized_pnl=100.0,
        positions=(PaperPosition("EURUSD", 0.1, 1.1, 100.0),),
    )

    store.save(state)

    assert store.load() == state
    assert not path.with_suffix(".json.tmp").exists()


def test_missing_state_returns_none(tmp_path) -> None:
    assert PaperRuntimeStateStore(tmp_path / "missing.json").load() is None
