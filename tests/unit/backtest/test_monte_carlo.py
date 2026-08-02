from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.broker import ClosedTrade
from fxbot.backtest.events import OrderSide
from fxbot.backtest.monte_carlo import (
    MonteCarloConfig,
    MonteCarloMethod,
    simulate_monte_carlo,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def trade(index: int, pnl: float) -> ClosedTrade:
    return ClosedTrade(
        trade_id=str(index),
        symbol="EURUSD",
        side=OrderSide.BUY,
        volume=1,
        entry_price=1.1,
        exit_price=1.101,
        opened_at=BASE,
        closed_at=BASE + timedelta(minutes=index + 1),
        gross_pnl=pnl,
        commission=0,
        net_pnl=pnl,
    )


def test_reshuffle_is_seeded_and_preserves_final_equity() -> None:
    trades = tuple(trade(i, pnl) for i, pnl in enumerate((100, -50, 25, -10)))
    config = MonteCarloConfig(iterations=20, seed=42, initial_equity=1_000)
    first = simulate_monte_carlo(trades, config=config)
    second = simulate_monte_carlo(trades, config=config)
    assert first == second
    assert {path.final_equity for path in first.paths} == {1_065}
    assert len(first.quantiles) == 3


def test_bootstrap_generates_outcome_distribution() -> None:
    trades = tuple(trade(i, pnl) for i, pnl in enumerate((100, -100, 50)))
    result = simulate_monte_carlo(
        trades,
        config=MonteCarloConfig(iterations=100, seed=2, initial_equity=1_000),
        method=MonteCarloMethod.BOOTSTRAP,
    )
    assert len({item.final_equity for item in result.paths}) > 1
    assert result.method is MonteCarloMethod.BOOTSTRAP


def test_risk_of_ruin_is_estimated_from_intrapathequity() -> None:
    trades = (trade(0, -600), trade(1, 300))
    result = simulate_monte_carlo(
        trades,
        config=MonteCarloConfig(
            iterations=100,
            seed=1,
            initial_equity=1_000,
            ruin_threshold_fraction=0.5,
        ),
    )
    assert 0.0 < result.risk_of_ruin < 1.0


def test_empty_trade_set_produces_flat_paths() -> None:
    result = simulate_monte_carlo(
        (),
        config=MonteCarloConfig(iterations=3, initial_equity=1_000),
    )
    assert all(item.final_equity == 1_000 for item in result.paths)
    assert result.risk_of_ruin == 0.0


def test_configuration_validation() -> None:
    with pytest.raises(ValueError, match="iterations"):
        MonteCarloConfig(iterations=0)
    with pytest.raises(ValueError, match="sorted"):
        MonteCarloConfig(confidence_levels=(0.95, 0.05))
