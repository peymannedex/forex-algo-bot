from __future__ import annotations

from typing import Any

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.liquidity import (
    LiquidityConfig,
    LiquidityPool,
    LiquiditySide,
    detect_liquidity_pools,
    detect_liquidity_sweeps,
    nearest_liquidity_target,
)
from fxbot.strategy.market_structure import StructureDirection, SwingKind, SwingPoint


def _swing(series: Any, index: int, kind: SwingKind, price: float) -> SwingPoint:
    return SwingPoint(
        "EURUSD",
        Timeframe.M5,
        index,
        kind,
        price,
        series.bars[index].open_time,
        series.bars[index + 1].close_time,
    )


def test_liquidity_config_validation() -> None:
    with pytest.raises(ValueError):
        LiquidityConfig(minimum_touches=1)
    with pytest.raises(ValueError):
        LiquidityConfig(sweep_buffer_atr=-1)


def test_detect_equal_high_and_low_pools(make_series_ohlc: Any) -> None:
    series = make_series_ohlc([(1.0, 1.2, 0.8, 1.0)] * 8)
    swings = (
        _swing(series, 1, SwingKind.HIGH, 1.2000),
        _swing(series, 3, SwingKind.HIGH, 1.2005),
        _swing(series, 2, SwingKind.LOW, 0.8000),
        _swing(series, 4, SwingKind.LOW, 0.8004),
    )

    pools = detect_liquidity_pools(series, swings, atr=0.01)

    assert {pool.side for pool in pools} == {
        LiquiditySide.BUY_SIDE,
        LiquiditySide.SELL_SIDE,
    }
    assert all(len(pool.swing_indices) == 2 for pool in pools)


def test_pool_is_invalidated_by_close_through(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.0, 1.2, 0.9, 1.0),
            (1.0, 1.2, 0.9, 1.0),
            (1.0, 1.2, 0.9, 1.0),
            (1.0, 1.25, 0.95, 1.22),
        ]
    )
    swings = (
        _swing(series, 0, SwingKind.HIGH, 1.20),
        _swing(series, 1, SwingKind.HIGH, 1.20),
    )

    pool = detect_liquidity_pools(series, swings, atr=0.01)[0]

    assert not pool.active
    assert pool.invalidated_at == series.bars[3].close_time


def test_detect_buy_side_sweep(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.22, 0.95, 1.19),
        ]
    )
    pool = LiquidityPool(
        "EURUSD",
        Timeframe.M5,
        LiquiditySide.BUY_SIDE,
        1.20,
        (0, 1),
        series.bars[2].close_time,
        0.001,
    )

    sweep = detect_liquidity_sweeps(series, (pool,), atr=0.01)[0]

    assert sweep.direction is StructureDirection.BEARISH
    assert sweep.penetration == pytest.approx(0.02)
    assert sweep.close_price == pytest.approx(1.19)


def test_detect_sell_side_sweep(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.05, 0.78, 0.81),
        ]
    )
    pool = LiquidityPool(
        "EURUSD",
        Timeframe.M5,
        LiquiditySide.SELL_SIDE,
        0.80,
        (0, 1),
        series.bars[2].close_time,
        0.001,
    )

    sweep = detect_liquidity_sweeps(series, (pool,), atr=0.01)[0]

    assert sweep.direction is StructureDirection.BULLISH
    assert sweep.extreme_price == pytest.approx(0.78)


def test_no_sweep_after_pool_invalidation(make_series_ohlc: Any) -> None:
    series = make_series_ohlc([(1.0, 1.3, 0.9, 1.2)] * 4)
    pool = LiquidityPool(
        "EURUSD",
        Timeframe.M5,
        LiquiditySide.BUY_SIDE,
        1.20,
        (0, 1),
        series.bars[1].close_time,
        0.001,
        invalidated_at=series.bars[2].close_time,
    )
    assert detect_liquidity_sweeps(series, (pool,), atr=0.01) == ()


def test_nearest_liquidity_target() -> None:
    from datetime import UTC, datetime

    formed = datetime(2026, 1, 1, tzinfo=UTC)
    low = LiquidityPool(
        "EURUSD", Timeframe.M5, LiquiditySide.SELL_SIDE, 1.0, (1, 2), formed, 0.001
    )
    near_high = LiquidityPool(
        "EURUSD", Timeframe.M5, LiquiditySide.BUY_SIDE, 1.2, (1, 2), formed, 0.001
    )
    far_high = LiquidityPool(
        "EURUSD", Timeframe.M5, LiquiditySide.BUY_SIDE, 1.4, (3, 4), formed, 0.001
    )

    assert nearest_liquidity_target(
        (low, near_high, far_high),
        direction=StructureDirection.BULLISH,
        price=1.1,
    ) is near_high
    assert nearest_liquidity_target(
        (low, near_high),
        direction=StructureDirection.BEARISH,
        price=1.1,
    ) is low
