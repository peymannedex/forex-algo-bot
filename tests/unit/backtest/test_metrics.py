from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.broker import BrokerSnapshot, ClosedTrade
from fxbot.backtest.config import BacktestConfig, InstrumentConfig
from fxbot.backtest.events import OrderSide
from fxbot.backtest.metrics import (
    calculate_performance,
    calendar_returns,
    compare_benchmark,
    equity_returns,
)
from fxbot.backtest.results import BacktestResult, EquityPoint

BASE = datetime(2025, 12, 31, tzinfo=UTC)


def point(index: int, equity: float, margin: float = 0.0) -> EquityPoint:
    return EquityPoint(
        BASE + timedelta(days=index),
        equity,
        equity,
        margin,
        equity - margin,
        0,
        0,
        0,
    )


def closed_trade(trade_id: str, pnl: float) -> ClosedTrade:
    return ClosedTrade(
        trade_id=trade_id,
        symbol="EURUSD",
        side=OrderSide.BUY,
        volume=1,
        entry_price=1.1,
        exit_price=1.101,
        opened_at=BASE,
        closed_at=BASE + timedelta(days=1),
        gross_pnl=pnl,
        commission=0,
        net_pnl=pnl,
    )


def result() -> BacktestResult:
    curve = (
        point(0, 10_000),
        point(1, 11_000, 1_000),
        point(2, 9_900, 1_000),
        point(366, 12_000),
    )
    snapshot = BrokerSnapshot(
        timestamp=curve[-1].timestamp,
        balance=12_000,
        equity=12_000,
        margin_used=0,
        free_margin=12_000,
        unrealized_pnl=0,
        realized_pnl=2_000,
        commissions=0,
        swap=0,
        positions=(),
        pending_orders=(),
    )
    return BacktestResult(
        config=BacktestConfig(
            initial_cash=10_000,
            instruments=(InstrumentConfig("EURUSD"),),
        ),
        started_at=curve[0].timestamp,
        ended_at=curve[-1].timestamp,
        final_snapshot=snapshot,
        orders=(),
        fills=(),
        trades=(closed_trade("1", 100), closed_trade("2", -50)),
        equity_curve=curve,
        audit_events=(),
    )


def test_equity_returns() -> None:
    values = equity_returns((point(0, 100), point(1, 110), point(2, 99)))
    assert values == pytest.approx((0.1, -0.1))


def test_calendar_returns_compound_by_month_and_year() -> None:
    curve = (
        point(0, 100),
        point(1, 110),
        point(32, 121),
        point(370, 133.1),
    )
    monthly = calendar_returns(curve, frequency="month")
    yearly = calendar_returns(curve, frequency="year")
    assert [item.period for item in monthly] == ["2026-01", "2026-02", "2027-01"]
    assert [item.return_fraction for item in monthly] == pytest.approx([0.1, 0.1, 0.1])
    assert [item.return_fraction for item in yearly] == pytest.approx([0.21, 0.1])


def test_performance_metrics_include_risk_and_trade_statistics() -> None:
    metrics = calculate_performance(result())
    assert metrics.net_profit == pytest.approx(2_000)
    assert metrics.total_return == pytest.approx(0.2)
    assert metrics.cagr == pytest.approx(0.2, rel=0.02)
    assert metrics.maximum_drawdown_fraction == pytest.approx(0.1)
    assert metrics.maximum_drawdown_amount == pytest.approx(1_100)
    assert metrics.trade_statistics.profit_factor == pytest.approx(2.0)
    assert metrics.exposure_fraction > 0.0
    assert metrics.time_in_market_seconds > 0.0


def test_invalid_frequency_and_non_positive_equity_are_rejected() -> None:
    with pytest.raises(ValueError, match="frequency"):
        calendar_returns((point(0, 100), point(1, 101)), frequency="week")
    bad = point(0, 0)
    with pytest.raises(ValueError, match="non-positive"):
        equity_returns((bad, point(1, 100)))


def test_benchmark_comparison_requires_synchronized_curves() -> None:
    strategy = (point(0, 100), point(1, 110), point(2, 121))
    benchmark = (point(0, 100), point(1, 105), point(2, 110.25))
    comparison = compare_benchmark(strategy, benchmark)
    assert comparison.strategy_total_return == pytest.approx(0.21)
    assert comparison.benchmark_total_return == pytest.approx(0.1025)
    assert comparison.excess_total_return == pytest.approx(0.1075)
    assert comparison.beta >= 0.0

    with pytest.raises(ValueError, match="equal length"):
        compare_benchmark(strategy, benchmark[:2])
