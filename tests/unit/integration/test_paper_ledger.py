from datetime import UTC, datetime, timedelta

import pytest

from fxbot.execution.models import ExecutionFill, OrderSide, Quote
from fxbot.integration.ledger import PaperPortfolioLedger

NOW = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)


def fill(
    identifier: str,
    side: OrderSide,
    quantity: float,
    price: float,
    *,
    commission: float = 0.0,
) -> ExecutionFill:
    return ExecutionFill(
        execution_id=identifier,
        broker_order_id=f"order-{identifier}",
        client_order_id=f"client-{identifier}",
        symbol="EURUSD",
        side=side,
        quantity=quantity,
        price=price,
        executed_at=NOW,
        commission=commission,
    )


def test_open_and_mark_long_position() -> None:
    ledger = PaperPortfolioLedger(initial_balance=100_000.0)
    ledger.on_quote(Quote("EURUSD", 1.1000, 1.1002, NOW))
    ledger.on_fill(fill("1", OrderSide.BUY, 0.1, 1.1002, commission=1.0))

    assert ledger.signed_position("EURUSD") == pytest.approx(0.1)
    assert ledger.balance == pytest.approx(99_999.0)
    assert ledger.unrealized_pnl == pytest.approx(-2.0)
    assert ledger.equity == pytest.approx(99_997.0)


def test_partial_close_realizes_profit() -> None:
    ledger = PaperPortfolioLedger(initial_balance=100_000.0)
    ledger.on_quote(Quote("EURUSD", 1.1000, 1.1002, NOW))
    ledger.on_fill(fill("1", OrderSide.BUY, 0.1, 1.1000))
    ledger.on_fill(fill("2", OrderSide.SELL, 0.04, 1.1010))

    assert ledger.signed_position("EURUSD") == pytest.approx(0.06)
    assert ledger.realized_pnl == pytest.approx(4.0)
    assert ledger.balance == pytest.approx(100_004.0)


def test_reversal_uses_new_fill_as_average_price() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1000, 1.1002, NOW))
    ledger.on_fill(fill("1", OrderSide.BUY, 0.1, 1.1000))
    ledger.on_fill(fill("2", OrderSide.SELL, 0.15, 1.1010))

    position = ledger.view().positions[0]
    assert position.signed_quantity == pytest.approx(-0.05)
    assert position.average_price == pytest.approx(1.1010)
    assert ledger.realized_pnl == pytest.approx(10.0)


def test_quotes_must_be_chronological() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, NOW))

    with pytest.raises(ValueError, match="chronological"):
        ledger.on_quote(
            Quote("EURUSD", 1.1, 1.1002, NOW - timedelta(seconds=1))
        )


def test_state_round_trip() -> None:
    ledger = PaperPortfolioLedger()
    ledger.on_quote(Quote("EURUSD", 1.1, 1.1002, NOW))
    ledger.on_fill(fill("1", OrderSide.BUY, 0.1, 1.1002))
    state = ledger.state(cycle=4, last_frame_at=NOW)

    restored = PaperPortfolioLedger()
    restored.restore(state)

    assert restored.balance == ledger.balance
    assert restored.signed_position("EURUSD") == pytest.approx(0.1)
