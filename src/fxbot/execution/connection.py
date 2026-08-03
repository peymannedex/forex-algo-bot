"""MetaTrader 5 terminal connection management with bounded recovery."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from threading import RLock
from time import sleep
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

from fxbot.execution.broker import PermanentBrokerError, TransientBrokerError

Sleeper = Callable[[float], None]
Clock = Callable[[], datetime]


@runtime_checkable
class MT5ExecutionClient(Protocol):
    """Structural contract for the official MetaTrader5 module and test fakes."""

    def initialize(self, *args: Any, **kwargs: Any) -> bool: ...

    def shutdown(self) -> None: ...

    def last_error(self) -> Any: ...

    def terminal_info(self) -> Any: ...

    def account_info(self) -> Any: ...

    def symbol_info(self, symbol: str) -> Any: ...

    def symbol_select(self, symbol: str, enable: bool) -> bool: ...

    def symbol_info_tick(self, symbol: str) -> Any: ...

    def order_check(self, request: dict[str, Any]) -> Any: ...

    def order_send(self, request: dict[str, Any]) -> Any: ...

    def orders_get(self, *args: Any, **kwargs: Any) -> Any: ...

    def history_orders_get(self, *args: Any, **kwargs: Any) -> Any: ...

    def history_deals_get(self, *args: Any, **kwargs: Any) -> Any: ...

    def positions_get(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class MT5ExecutionConnectionConfig:
    """Credentials and terminal options forwarded to ``MetaTrader5.initialize``."""

    terminal_path: Path | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 60_000
    portable: bool = False
    require_terminal_connected: bool = True
    require_trade_allowed: bool = True

    def __post_init__(self) -> None:
        if self.login is not None and self.login <= 0:
            raise ValueError("login must be positive")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    def initialize_args(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        positional: tuple[Any, ...] = (
            (str(self.terminal_path),) if self.terminal_path is not None else ()
        )
        keyword: dict[str, Any] = {"timeout": self.timeout_ms, "portable": self.portable}
        if self.login is not None:
            keyword["login"] = self.login
        if self.password is not None:
            keyword["password"] = self.password
        if self.server is not None:
            keyword["server"] = self.server
        return positional, keyword

    def __repr__(self) -> str:
        return (
            "MT5ExecutionConnectionConfig("
            f"terminal_path={self.terminal_path!r}, login={self.login!r}, "
            f"password={'***' if self.password is not None else None!r}, "
            f"server={self.server!r}, timeout_ms={self.timeout_ms!r}, "
            f"portable={self.portable!r}, "
            f"require_terminal_connected={self.require_terminal_connected!r}, "
            f"require_trade_allowed={self.require_trade_allowed!r})"
        )


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """Bounded exponential reconnect policy."""

    max_attempts: int = 4
    initial_delay_seconds: float = 0.5
    multiplier: float = 2.0
    maximum_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        for name in ("initial_delay_seconds", "multiplier", "maximum_delay_seconds"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be at least one")

    def delay_before_attempt(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        if attempt == 1:
            return 0.0
        raw = self.initial_delay_seconds * self.multiplier ** (attempt - 2)
        return min(raw, self.maximum_delay_seconds)


@dataclass(frozen=True, slots=True)
class MT5ConnectionSnapshot:
    """Auditable connection state returned by the manager."""

    connected: bool
    account_login: int | None
    server: str | None
    trade_allowed: bool
    checked_at: datetime


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


class MT5ConnectionManager:
    """Own and recover one local MT5 Python bridge connection."""

    def __init__(
        self,
        *,
        config: MT5ExecutionConnectionConfig | None = None,
        reconnect_policy: ReconnectPolicy | None = None,
        client: MT5ExecutionClient | None = None,
        sleeper: Sleeper = sleep,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or MT5ExecutionConnectionConfig()
        self.reconnect_policy = reconnect_policy or ReconnectPolicy()
        self._client = client
        self._sleeper = sleeper
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connected = False
        self._lock = RLock()

    @property
    def client(self) -> MT5ExecutionClient:
        if self._client is None:
            try:
                module: ModuleType = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise PermanentBrokerError(
                    "MetaTrader5 package is not installed in this Python environment"
                ) from exc
            self._client = cast(MT5ExecutionClient, module)
        return self._client

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> MT5ConnectionSnapshot:
        with self._lock:
            if self._connected:
                return self.snapshot()
            args, kwargs = self.config.initialize_args()
            if not self.client.initialize(*args, **kwargs):
                raise TransientBrokerError(f"MT5 initialize failed: {self.client.last_error()!r}")
            try:
                snapshot = self.snapshot(require_connected=False)
                self._validate_snapshot(snapshot)
            except Exception:
                self.client.shutdown()
                raise
            self._connected = True
            return snapshot

    def disconnect(self) -> None:
        with self._lock:
            if self._connected:
                self.client.shutdown()
            self._connected = False

    def ensure_connected(self) -> MT5ConnectionSnapshot:
        if not self._connected:
            return self.connect()
        snapshot = self.snapshot()
        try:
            self._validate_snapshot(snapshot)
        except TransientBrokerError:
            return self.reconnect()
        return snapshot

    def reconnect(self) -> MT5ConnectionSnapshot:
        last_error: Exception | None = None
        for attempt in range(1, self.reconnect_policy.max_attempts + 1):
            delay = self.reconnect_policy.delay_before_attempt(attempt)
            if delay > 0.0:
                self._sleeper(delay)
            try:
                self.disconnect()
                return self.connect()
            except TransientBrokerError as exc:
                last_error = exc
        raise TransientBrokerError(
            f"MT5 reconnect failed after {self.reconnect_policy.max_attempts} attempts"
        ) from last_error

    def snapshot(self, *, require_connected: bool = True) -> MT5ConnectionSnapshot:
        if require_connected and not self._connected:
            raise TransientBrokerError("MT5 connection is not initialized")
        terminal = self.client.terminal_info()
        account = self.client.account_info()
        if terminal is None or account is None:
            raise TransientBrokerError(f"MT5 connection health check failed: {self.client.last_error()!r}")
        terminal_connected = bool(_field(terminal, "connected", True))
        terminal_trade = bool(_field(terminal, "trade_allowed", True))
        account_trade = bool(
            _field(account, "trade_allowed", _field(account, "trade_expert", True))
        )
        return MT5ConnectionSnapshot(
            connected=terminal_connected,
            account_login=(
                int(_field(account, "login")) if _field(account, "login", None) is not None else None
            ),
            server=(
                str(_field(account, "server")) if _field(account, "server", None) is not None else None
            ),
            trade_allowed=terminal_trade and account_trade,
            checked_at=self._clock(),
        )

    def ensure_symbol(self, symbol: str) -> Any:
        self.ensure_connected()
        normalized = symbol.strip().upper()
        if not normalized:
            raise PermanentBrokerError("symbol cannot be empty")
        info = self.client.symbol_info(normalized)
        if info is None:
            raise PermanentBrokerError(f"MT5 symbol not found: {normalized}")
        if not bool(_field(info, "visible", True)):
            if not self.client.symbol_select(normalized, True):
                raise PermanentBrokerError(
                    f"MT5 could not enable symbol {normalized}: {self.client.last_error()!r}"
                )
            info = self.client.symbol_info(normalized)
            if info is None:
                raise PermanentBrokerError(f"MT5 symbol unavailable after selection: {normalized}")
        return info

    def symbol_tick(self, symbol: str) -> Any:
        self.ensure_symbol(symbol)
        tick = self.client.symbol_info_tick(symbol.strip().upper())
        if tick is None:
            raise TransientBrokerError(
                f"MT5 quote unavailable for {symbol}: {self.client.last_error()!r}"
            )
        return tick

    def _validate_snapshot(self, snapshot: MT5ConnectionSnapshot) -> None:
        if self.config.require_terminal_connected and not snapshot.connected:
            raise TransientBrokerError("MT5 terminal reports disconnected")
        if self.config.require_trade_allowed and not snapshot.trade_allowed:
            raise PermanentBrokerError("MT5 terminal or account does not allow algorithmic trading")


__all__ = [
    "MT5ConnectionManager",
    "MT5ConnectionSnapshot",
    "MT5ExecutionClient",
    "MT5ExecutionConnectionConfig",
    "ReconnectPolicy",
]
