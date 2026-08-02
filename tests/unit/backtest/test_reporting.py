from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.broker import BrokerSnapshot, ClosedTrade
from fxbot.backtest.config import BacktestConfig, InstrumentConfig
from fxbot.backtest.events import OrderSide
from fxbot.backtest.monte_carlo import MonteCarloConfig
from fxbot.backtest.reporting import PerformanceReport, build_performance_report
from fxbot.backtest.results import BacktestResult, EquityPoint

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def trade(
    trade_id: str,
    symbol: str,
    side: OrderSide,
    pnl: float,
) -> ClosedTrade:
    return ClosedTrade(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        volume=1,
        entry_price=1.1,
        exit_price=1.101,
        opened_at=BASE,
        closed_at=BASE + timedelta(days=1),
        gross_pnl=pnl,
        commission=0,
        net_pnl=pnl,
    )


def backtest_result() -> BacktestResult:
    curve = (
        EquityPoint(BASE, 10_000, 10_000, 0, 10_000, 0, 0, 0),
        EquityPoint(
            BASE + timedelta(days=1),
            10_100,
            10_100,
            0,
            10_100,
            0,
            0,
            0,
        ),
    )
    snapshot = BrokerSnapshot(
        timestamp=curve[-1].timestamp,
        balance=10_100,
        equity=10_100,
        margin_used=0,
        free_margin=10_100,
        unrealized_pnl=0,
        realized_pnl=100,
        commissions=0,
        swap=0,
        positions=(),
        pending_orders=(),
    )
    return BacktestResult(
        config=BacktestConfig(
            initial_cash=10_000,
            instruments=(InstrumentConfig("EURUSD"), InstrumentConfig("GBPUSD")),
        ),
        started_at=curve[0].timestamp,
        ended_at=curve[-1].timestamp,
        final_snapshot=snapshot,
        orders=(),
        fills=(),
        trades=(
            trade("1", "EURUSD", OrderSide.BUY, 150),
            trade("2", "GBPUSD", OrderSide.SELL, -50),
        ),
        equity_curve=curve,
        audit_events=(),
    )


def test_report_contains_breakdowns_and_monte_carlo() -> None:
    report = build_performance_report(
        backtest_result(),
        monte_carlo_config=MonteCarloConfig(iterations=10, initial_equity=10_000),
    )
    assert [item.key for item in report.symbol_breakdown] == ["EURUSD", "GBPUSD"]
    assert [item.key for item in report.side_breakdown] == ["buy", "sell"]
    assert report.monte_carlo is not None
    assert report.strategy_breakdown == ()
    assert report.metrics.net_profit == pytest.approx(100)


def test_report_json_and_fingerprint_are_stable() -> None:
    first = build_performance_report(backtest_result())
    second = build_performance_report(backtest_result())
    assert first.fingerprint == second.fingerprint
    assert first.to_json().startswith("{")
    assert first.to_dict()["generated_at"].endswith("+00:00")


def test_report_requires_timezone_aware_generation_time() -> None:
    base = build_performance_report(backtest_result())
    with pytest.raises(ValueError, match="timezone-aware"):
        PerformanceReport(
            generated_at=datetime(2026, 1, 1),
            result_fingerprint=base.result_fingerprint,
            metrics=base.metrics,
            drawdowns=base.drawdowns,
            monthly_returns=base.monthly_returns,
            yearly_returns=base.yearly_returns,
            symbol_breakdown=base.symbol_breakdown,
            side_breakdown=base.side_breakdown,
            strategy_breakdown=base.strategy_breakdown,
        )


def test_report_supports_strategy_and_benchmark_breakdowns() -> None:
    source = backtest_result()
    report = build_performance_report(
        source,
        strategy_by_trade_id={"1": "trend", "2": "mean_reversion"},
        benchmark_curve=(
            EquityPoint(BASE, 10_000, 10_000, 0, 10_000, 0, 0, 0),
            EquityPoint(
                BASE + timedelta(days=1),
                10_050,
                10_050,
                0,
                10_050,
                0,
                0,
                0,
            ),
        ),
    )
    assert [item.key for item in report.strategy_breakdown] == [
        "mean_reversion",
        "trend",
    ]
    assert report.benchmark is not None
    assert report.benchmark.excess_total_return == pytest.approx(0.005)


def test_strategy_mapping_rejects_unknown_trades() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_performance_report(
            backtest_result(),
            strategy_by_trade_id={"missing": "trend"},
        )
