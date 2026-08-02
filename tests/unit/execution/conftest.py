from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.execution.models import OrderIntent, OrderSide, OrderType, Quote

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
