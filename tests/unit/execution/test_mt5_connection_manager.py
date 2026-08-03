from pathlib import Path
from typing import Any

import pytest

from fxbot.execution.broker import PermanentBrokerError, TransientBrokerError
from fxbot.execution.connection import (
    MT5ConnectionManager,
    MT5ExecutionConnectionConfig,
    ReconnectPolicy,
)


def test_config_redacts_password() -> None:
    config = MT5ExecutionConnectionConfig(
        terminal_path=Path("terminal.exe"),
        login=123,
        password="secret",
        server="Demo",
    )
    assert "secret" not in repr(config)
    assert "***" in repr(config)


def test_initialize_arguments() -> None:
    config = MT5ExecutionConnectionConfig(
        terminal_path=Path("x"),
        login=123,
        password="pw",
        server="srv",
        timeout_ms=1234,
        portable=True,
    )
    args, kwargs = config.initialize_args()

    assert args == ("x",)
    assert kwargs["login"] == 123
    assert kwargs["portable"] is True


def test_connect_and_snapshot(client: Any) -> None:
    manager = MT5ConnectionManager(client=client)
    snapshot = manager.connect()

    assert manager.connected
    assert snapshot.account_login == 123456
    assert snapshot.trade_allowed


def test_connect_is_idempotent(client: Any) -> None:
    manager = MT5ConnectionManager(client=client)

    manager.connect()
    manager.connect()

    assert client.initialize_calls == 1


def test_initialize_failure_is_transient(client: Any) -> None:
    client.initialize_result = False

    with pytest.raises(TransientBrokerError):
        MT5ConnectionManager(client=client).connect()


def test_trade_disabled_is_permanent(client: Any) -> None:
    client.trade_allowed = False

    with pytest.raises(PermanentBrokerError, match="does not allow"):
        MT5ConnectionManager(client=client).connect()

    assert client.shutdown_calls == 1


def test_symbol_selection(client: Any) -> None:
    client.info.visible = False
    manager = MT5ConnectionManager(client=client)
    manager.connect()

    assert manager.ensure_symbol("EURUSD") is client.info


def test_unknown_symbol_rejected(client: Any) -> None:
    manager = MT5ConnectionManager(client=client)
    manager.connect()

    with pytest.raises(PermanentBrokerError, match="not found"):
        manager.ensure_symbol("GBPUSD")


def test_reconnect_uses_bounded_delays(client: Any) -> None:
    delays: list[float] = []
    manager = MT5ConnectionManager(
        client=client,
        reconnect_policy=ReconnectPolicy(
            max_attempts=3,
            initial_delay_seconds=0.1,
            multiplier=2.0,
            maximum_delay_seconds=1.0,
        ),
        sleeper=delays.append,
    )
    manager.connect()
    client.terminal_connected = False

    calls = 0
    original_initialize = client.initialize

    def initialize(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        client.terminal_connected = calls >= 2
        return original_initialize(*args, **kwargs)

    client.initialize = initialize

    snapshot = manager.reconnect()

    assert snapshot.connected
    assert delays == [0.1]


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        ReconnectPolicy(max_attempts=0)

    assert ReconnectPolicy().delay_before_attempt(1) == 0.0
