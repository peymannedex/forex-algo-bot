from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from fxbot.execution.models import OrderIntent, OrderSide, OrderType, Quote


@dataclass
class FakeClient:
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 8

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7

    ORDER_TIME_GTC = 0
    ORDER_TIME_DAY = 1
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2

    ORDER_STATE_STARTED = 0
    ORDER_STATE_PLACED = 1
    ORDER_STATE_CANCELED = 2
    ORDER_STATE_PARTIAL = 3
    ORDER_STATE_FILLED = 4
    ORDER_STATE_REJECTED = 5
    ORDER_STATE_EXPIRED = 6
    ORDER_STATE_REQUEST_ADD = 7
    ORDER_STATE_REQUEST_MODIFY = 8
    ORDER_STATE_REQUEST_CANCEL = 9

    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_ERROR = 10011
    TRADE_RETCODE_TIMEOUT = 10012
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_TRADE_DISABLED = 10017
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
    TRADE_RETCODE_NO_CHANGES = 10025
    TRADE_RETCODE_LOCKED = 10028
    TRADE_RETCODE_CONNECTION = 10031

    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1

    initialized: bool = False
    initialize_result: bool = True
    terminal_connected: bool = True
    trade_allowed: bool = True
    selected: bool = True
    check_result: Any = None
    send_result: Any = None

    def __post_init__(self) -> None:
        self.info = SimpleNamespace(
            visible=True,
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_stops_level=10,
            trade_freeze_level=5,
            filling_mode=self.ORDER_FILLING_RETURN,
        )
        self.tick = SimpleNamespace(
            bid=1.1000,
            ask=1.1002,
            time_msc=1704067200000,
        )
        self.open_orders: list[Any] = []
        self.history_orders: list[Any] = []
        self.deals: list[Any] = []
        self.positions: list[Any] = []
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_calls = 0
        self.initialize_calls = 0

        if self.check_result is None:
            self.check_result = SimpleNamespace(retcode=0, comment="ok")
        if self.send_result is None:
            self.send_result = SimpleNamespace(
                retcode=self.TRADE_RETCODE_DONE,
                order=101,
                deal=501,
                volume=0.1,
                price=1.1002,
                comment="done",
            )

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        self.initialize_calls += 1
        self.initialized = self.initialize_result
        return self.initialize_result

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.initialized = False

    def last_error(self) -> tuple[int, str]:
        return (1, "fake")

    def terminal_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            connected=self.terminal_connected,
            trade_allowed=self.trade_allowed,
        )

    def account_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            login=123456,
            server="Demo",
            trade_allowed=self.trade_allowed,
        )

    def symbol_info(self, symbol: str) -> Any:
        return self.info if symbol == "EURUSD" else None

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return self.selected

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace:
        return self.tick

    def order_check(self, request: dict[str, Any]) -> Any:
        self.requests.append(("check", request))
        return self.check_result

    def order_send(self, request: dict[str, Any]) -> Any:
        self.requests.append(("send", request))
        return self.send_result

    def orders_get(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if "ticket" in kwargs:
            return tuple(
                item for item in self.open_orders if item.ticket == kwargs["ticket"]
            )
        return tuple(self.open_orders)

    def history_orders_get(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if "ticket" in kwargs:
            return tuple(
                item for item in self.history_orders if item.ticket == kwargs["ticket"]
            )
        return tuple(self.history_orders)

    def history_deals_get(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return tuple(self.deals)

    def positions_get(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return tuple(self.positions)


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


BASE = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


@pytest.fixture
def market_intent() -> OrderIntent:
    return OrderIntent(
        client_order_id="client-1",
        idempotency_key="key-1",
        symbol="eurusd",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
        created_at=BASE,
        strategy_id="trend",
        metadata=(("signal", "buy"),),
    )


@pytest.fixture
def quote() -> Quote:
    return Quote("EURUSD", 1.1000, 1.1002, BASE + timedelta(seconds=1))
