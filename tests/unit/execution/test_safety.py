from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxbot.execution.broker import PermanentBrokerError
from fxbot.execution.safety import ExecutionControl, ExecutionControlState

BASE = datetime(2026, 1, 5, tzinfo=UTC)


def test_control_arm_trip_and_guard() -> None:
    control = ExecutionControl.armed(clock=lambda: BASE)
    control.ensure_enabled()
    state = control.trip("daily loss limit")
    assert not state.enabled
    with pytest.raises(PermanentBrokerError, match="daily loss limit"):
        control.ensure_enabled()
    assert control.arm().enabled


def test_disarmed_factory() -> None:
    control = ExecutionControl.disarmed("maintenance", clock=lambda: BASE)
    assert control.state.reason == "maintenance"


def test_control_state_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ExecutionControlState(True, "x", datetime(2026, 1, 1))
