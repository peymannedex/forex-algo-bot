from datetime import UTC, datetime, time, timedelta

import pytest

from fxbot.execution.models import (
    ExecutionFill,
    OrderIntent,
    OrderSide,
    OrderType,
    Quote,
)
from fxbot.execution.safety import ExecutionControl
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.risk import GuardedPaperRiskAuthorizer, PaperExposureLimits
from fxbot.production.protections import (
    LossGuard,
    MarketHoursGuard,
    MarketWindow,
    QuoteGuard,
    QuoteProtectionConfig,
)

NOW = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


def intent(
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: float = 0.2,
    reduce_only: bool = False,
    created_at: datetime = NOW,
) -> OrderIntent:
    return OrderIntent(
        client_order_id=f"client-{side.value}-{quantity}-{reduce_only}",
        idempotency_key=f"key-{side.value}-{quantity}-{reduce_only}",
        symbol="EURUSD",
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        created_at=created_at,
        reduce_only=reduce_only,
    )


def authorizer(
    ledger: PaperPortfolioLedger,
    *,
    max_position: float = 1.0,
) -> GuardedPaperRiskAuthorizer:
    control = ExecutionControl.armed()
    return GuardedPaperRiskAuthorizer(
        ledger=ledger,
        quote_guard=QuoteGuard(QuoteProtectionConfig(3.0, 20.0)),
        loss_guard=LossGuard(control, max_daily_loss=500.0, max_drawdown=1_000.0),
        market_hours_guard=MarketHoursGuard(
            (MarketWindow(frozenset({0, 1, 2, 3, 4}), time(0), time(23, 59)),)
        ),
        limits=PaperExposureLimits(max_position, 3.0),
    )


def test_rejects_missing_quote() -> None:
    decision = authorizer(PaperPortfolioLedger()).authorize(intent())

    assert not decision.approved
    assert "quote" in decision.reason


def test_rejects_stale_quote() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, NOW - timedelta(seconds=10)))

    decision = authorizer(ledger).authorize(intent())

    assert not decision.approved
    assert "stale" in decision.reason


def test_reduces_quantity_to_symbol_capacity() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, NOW))
    ledger.on_fill(
        ExecutionFill(
            "fill-1",
            "order-1",
            "client-1",
            "EURUSD",
            OrderSide.BUY,
            0.8,
            1.1002,
            NOW,
        )
    )

    decision = authorizer(ledger, max_position=1.0).authorize(intent(quantity=0.5))

    assert decision.approved
    assert decision.approved_quantity == pytest.approx(0.2)


def test_reduce_only_requires_opposite_existing_position() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, NOW))

    decision = authorizer(ledger).authorize(
        intent(side=OrderSide.SELL, reduce_only=True)
    )

    assert not decision.approved
    assert "reduce" in decision.reason


def test_reduce_only_is_capped_to_existing_quantity() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, NOW))
    ledger.on_fill(
        ExecutionFill(
            "fill-1",
            "order-1",
            "client-1",
            "EURUSD",
            OrderSide.BUY,
            0.1,
            1.1002,
            NOW,
        )
    )

    decision = authorizer(ledger).authorize(
        intent(side=OrderSide.SELL, quantity=0.4, reduce_only=True)
    )

    assert decision.approved
    assert decision.approved_quantity == pytest.approx(0.1)


def test_weekend_entry_is_rejected() -> None:
    weekend = datetime(2026, 1, 10, 14, 0, tzinfo=UTC)
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, weekend))

    decision = authorizer(ledger).authorize(intent(created_at=weekend))

    assert not decision.approved
    assert "closed" in decision.reason
