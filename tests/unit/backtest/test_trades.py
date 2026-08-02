from datetime import UTC, datetime, timedelta
from math import inf

import pytest

from fxbot.backtest.broker import ClosedTrade
from fxbot.backtest.events import OrderSide
from fxbot.backtest.trades import TradeExcursion, analyze_trades

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def trade(
    trade_id: str,
    pnl: float,
    *,
    minutes: int = 10,
    symbol: str = "EURUSD",
    side: OrderSide = OrderSide.BUY,
) -> ClosedTrade:
    return ClosedTrade(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        volume=1.0,
        entry_price=1.1,
        exit_price=1.101,
        opened_at=BASE,
        closed_at=BASE + timedelta(minutes=minutes),
        gross_pnl=pnl + 2.0,
        commission=2.0,
        net_pnl=pnl,
    )


def test_trade_statistics_profitability_and_streaks() -> None:
    stats = analyze_trades(
        (
            trade("1", 100),
            trade("2", 50),
            trade("3", -25),
            trade("4", -75),
            trade("5", 0),
        )
    )
    assert stats.trade_count == 5
    assert stats.winning_trades == 2
    assert stats.losing_trades == 2
    assert stats.breakeven_trades == 1
    assert stats.win_rate == pytest.approx(0.4)
    assert stats.gross_profit == pytest.approx(150)
    assert stats.gross_loss == pytest.approx(100)
    assert stats.net_profit == pytest.approx(50)
    assert stats.expectancy == pytest.approx(10)
    assert stats.payoff_ratio == pytest.approx(1.5)
    assert stats.profit_factor == pytest.approx(1.5)
    assert stats.maximum_consecutive_wins == 2
    assert stats.maximum_consecutive_losses == 2
    assert stats.total_commission == pytest.approx(10)


def test_trade_statistics_empty_and_one_sided_ledgers() -> None:
    empty = analyze_trades(())
    assert empty.trade_count == 0
    assert empty.profit_factor == 0.0

    winners = analyze_trades((trade("1", 10), trade("2", 20)))
    assert winners.profit_factor == inf
    assert winners.payoff_ratio == inf


def test_trade_holding_time_and_excursions() -> None:
    stats = analyze_trades(
        (trade("1", 10, minutes=10), trade("2", -5, minutes=20)),
        excursions=(
            TradeExcursion("1", 4, 15),
            TradeExcursion("2", 8, 3),
        ),
    )
    assert stats.average_holding_seconds == pytest.approx(900)
    assert stats.median_holding_seconds == pytest.approx(900)
    assert stats.average_mae == pytest.approx(6)
    assert stats.average_mfe == pytest.approx(9)


def test_excursions_must_be_unique_and_reference_known_trades() -> None:
    with pytest.raises(ValueError, match="unique"):
        analyze_trades(
            (trade("1", 10),),
            excursions=(TradeExcursion("1", 1, 2), TradeExcursion("1", 1, 2)),
        )
    with pytest.raises(ValueError, match="unknown"):
        analyze_trades(
            (trade("1", 10),),
            excursions=(TradeExcursion("other", 1, 2),),
        )


def test_trade_excursion_validation() -> None:
    with pytest.raises(ValueError, match="trade_id"):
        TradeExcursion(" ", 1, 2)
    with pytest.raises(ValueError, match="non-negative"):
        TradeExcursion("x", -1, 2)
