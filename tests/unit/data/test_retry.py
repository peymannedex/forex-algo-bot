from __future__ import annotations

import pytest

from fxbot.data.retry import ReconnectPolicy


def test_reconnect_policy_validates_configuration() -> None:
    with pytest.raises(ValueError, match="initial_delay"):
        ReconnectPolicy(initial_delay_seconds=0)
    with pytest.raises(ValueError, match="max_delay"):
        ReconnectPolicy(initial_delay_seconds=2, max_delay_seconds=1)
    with pytest.raises(ValueError, match="multiplier"):
        ReconnectPolicy(multiplier=0.5)
    with pytest.raises(ValueError, match="jitter"):
        ReconnectPolicy(jitter_ratio=1.1)
    with pytest.raises(ValueError, match="max_attempts"):
        ReconnectPolicy(max_attempts=-1)


def test_reconnect_policy_calculates_bounded_jittered_delay() -> None:
    policy = ReconnectPolicy(
        initial_delay_seconds=2,
        max_delay_seconds=10,
        multiplier=2,
        jitter_ratio=0.25,
        max_attempts=2,
    )

    assert policy.delay_seconds(1, random_value=0.5) == 2
    assert policy.delay_seconds(2, random_value=0.0) == 3
    assert policy.delay_seconds(4, random_value=1.0) == 12.5
    assert policy.allows_retry(0)
    assert policy.allows_retry(1)
    assert not policy.allows_retry(2)

    with pytest.raises(ValueError, match="attempt"):
        policy.delay_seconds(0)
    with pytest.raises(ValueError, match="random_value"):
        policy.delay_seconds(1, random_value=2)
    with pytest.raises(ValueError, match="completed_attempts"):
        policy.allows_retry(-1)
