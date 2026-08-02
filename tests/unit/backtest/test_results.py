from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.broker import BrokerSnapshot
from fxbot.backtest.config import BacktestConfig, InstrumentConfig
from fxbot.backtest.results import BacktestResult, EquityPoint, equity_point

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def snapshot(timestamp: datetime, equity: float) -> BrokerSnapshot:
    return BrokerSnapshot(
        timestamp=timestamp,
        balance=equity,
        equity=equity,
        margin_used=0.0,
        free_margin=equity,
        unrealized_pnl=0.0,
        realized_pnl=equity - 10_000.0,
        commissions=0.0,
        swap=0.0,
        positions=(),
        pending_orders=(),
    )


def config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=10_000,
        instruments=(InstrumentConfig("EURUSD"),),
    )


def test_equity_point_calculates_drawdown_from_running_peak() -> None:
    point = equity_point(snapshot(BASE, 9_000), 10_000)
    assert point.drawdown_amount == pytest.approx(1_000)
    assert point.drawdown_fraction == pytest.approx(0.1)


def test_result_return_drawdown_and_counts() -> None:
    points = (
        EquityPoint(BASE, 10_000, 10_000, 0, 10_000, 0, 0, 0),
        EquityPoint(
            BASE + timedelta(seconds=1),
            9_000,
            9_000,
            0,
            9_000,
            0,
            1_000,
            0.1,
        ),
        EquityPoint(
            BASE + timedelta(seconds=2),
            11_000,
            11_000,
            0,
            11_000,
            0,
            0,
            0,
        ),
    )
    result = BacktestResult(
        config=config(),
        started_at=BASE,
        ended_at=BASE + timedelta(seconds=2),
        final_snapshot=snapshot(BASE + timedelta(seconds=2), 11_000),
        orders=(),
        fills=(),
        trades=(),
        equity_curve=points,
        audit_events=(),
    )
    assert result.total_return == pytest.approx(0.1)
    assert result.max_drawdown_fraction == pytest.approx(0.1)
    assert result.max_drawdown_amount == pytest.approx(1_000)


def test_result_fingerprint_is_stable() -> None:
    kwargs = dict(
        config=config(),
        started_at=BASE,
        ended_at=BASE,
        final_snapshot=snapshot(BASE, 10_000),
        orders=(),
        fills=(),
        trades=(),
        equity_curve=(
            EquityPoint(BASE, 10_000, 10_000, 0, 10_000, 0, 0, 0),
        ),
        audit_events=(),
    )
    first = BacktestResult(**kwargs)
    second = BacktestResult(**kwargs)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_result_rejects_out_of_order_equity_curve() -> None:
    with pytest.raises(ValueError, match="chronologically"):
        BacktestResult(
            config=config(),
            started_at=BASE,
            ended_at=BASE + timedelta(seconds=2),
            final_snapshot=snapshot(BASE + timedelta(seconds=2), 10_000),
            orders=(),
            fills=(),
            trades=(),
            equity_curve=(
                EquityPoint(
                    BASE + timedelta(seconds=2),
                    10_000,
                    10_000,
                    0,
                    10_000,
                    0,
                    0,
                    0,
                ),
                EquityPoint(BASE, 10_000, 10_000, 0, 10_000, 0, 0, 0),
            ),
            audit_events=(),
        )
