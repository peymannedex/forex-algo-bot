from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isclose

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.strategy.context import MarketSeries
from fxbot.strategy.indicators import (
    IndicatorConfig,
    IndicatorError,
    RollingIndicatorState,
    average_spread,
    average_true_range,
    calculate_indicators,
    exponential_moving_average,
    momentum,
    realized_volatility,
    relative_strength_index,
    simple_moving_average,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_bar(index: int, close: float, *, spread: float = 0.0002) -> Bar:
    bid_close = close - spread
    return Bar(
        symbol="EURUSD",
        open_time=BASE + timedelta(minutes=5 * index),
        timeframe=Timeframe.M5,
        bid=OHLC(bid_close, bid_close + 0.0005, bid_close - 0.0005, bid_close),
        ask=OHLC(close, close + 0.0005, close - 0.0005, close),
    )


def test_simple_and_exponential_moving_average() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert simple_moving_average(values, 3) == 4.0
    assert exponential_moving_average(values, 3) == 4.0


def test_rsi_handles_up_down_and_flat_series() -> None:
    assert relative_strength_index([1, 2, 3, 4, 5, 6], 3) == 100.0
    assert relative_strength_index([6, 5, 4, 3, 2, 1], 3) == 0.0
    assert relative_strength_index([1, 1, 1, 1, 1], 3) == 50.0


def test_momentum_returns_fractional_change() -> None:
    assert isclose(momentum([100.0, 101.0, 102.0], 2), 0.02)


def test_realized_volatility_is_zero_for_constant_prices() -> None:
    assert realized_volatility([1.0] * 6, 5) == 0.0


def test_atr_and_spread() -> None:
    bars = tuple(make_bar(i, 1.10 + i * 0.0001) for i in range(20))
    assert average_true_range(bars, 14) > 0.0
    assert isclose(average_spread(bars, 10), 0.0002, abs_tol=1e-12)


def test_indicator_config_validates_periods() -> None:
    with pytest.raises(ValueError, match="fast_ema"):
        IndicatorConfig(fast_ema_period=26, slow_ema_period=12)
    with pytest.raises(ValueError, match="positive"):
        IndicatorConfig(atr_period=0)


def test_calculate_indicators_returns_canonical_values() -> None:
    config = IndicatorConfig(
        fast_ema_period=3,
        slow_ema_period=5,
        sma_period=4,
        atr_period=3,
        rsi_period=3,
        momentum_period=3,
        volatility_period=4,
        spread_period=4,
    )
    bars = tuple(make_bar(i, 1.10 + i * 0.0002) for i in range(10))
    snapshot = calculate_indicators(MarketSeries("EURUSD", Timeframe.M5, bars), config)
    assert snapshot.value("fast_ema") > snapshot.value("slow_ema")
    assert snapshot.value("rsi") == 100.0
    assert snapshot.value("atr") > 0.0
    assert snapshot.value("spread_to_atr") > 0.0


def test_calculate_indicators_rejects_insufficient_history() -> None:
    series = MarketSeries("EURUSD", Timeframe.M5, (make_bar(0, 1.1),))
    with pytest.raises(IndicatorError, match="requires"):
        calculate_indicators(series)


def test_rolling_state_rejects_wrong_or_out_of_order_bars() -> None:
    state = RollingIndicatorState("EURUSD", Timeframe.M5, max_bars=5)
    state.append(make_bar(0, 1.1))
    with pytest.raises(ValueError, match="strictly"):
        state.append(make_bar(0, 1.1))
    with pytest.raises(ValueError, match="symbol"):
        wrong = make_bar(1, 1.1)
        object.__setattr__(wrong, "symbol", "GBPUSD")
        state.append(wrong)


def test_rolling_state_is_bounded_and_snapshots() -> None:
    config = IndicatorConfig(
        fast_ema_period=2,
        slow_ema_period=3,
        sma_period=2,
        atr_period=2,
        rsi_period=2,
        momentum_period=2,
        volatility_period=2,
        spread_period=2,
    )
    state = RollingIndicatorState("EURUSD", Timeframe.M5, max_bars=4)
    state.extend(make_bar(i, 1.1 + i * 0.0001) for i in range(6))
    assert len(state.bars) == 4
    assert state.snapshot(config).sample_size == 4


@pytest.mark.parametrize(
    "call",
    [
        lambda: simple_moving_average([1.0], 2),
        lambda: average_true_range([make_bar(0, 1.1)], 2),
        lambda: momentum([0.0, 1.0], 1),
        lambda: realized_volatility([0.0, 1.0], 1),
    ],
)
def test_indicator_errors_are_explicit(call: object) -> None:
    with pytest.raises(IndicatorError):
        call()  # type: ignore[operator]
