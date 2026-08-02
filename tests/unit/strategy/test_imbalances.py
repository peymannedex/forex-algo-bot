from __future__ import annotations

from typing import Any

import pytest

from fxbot.strategy.imbalances import (
    ImbalanceConfig,
    ImbalanceSide,
    detect_fair_value_gaps,
    nearest_fair_value_gap,
)
from fxbot.strategy.market_structure import StructureDirection


def test_imbalance_config_validation() -> None:
    with pytest.raises(ValueError):
        ImbalanceConfig(minimum_gap_atr=-1)


def test_detect_bullish_fair_value_gap(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.15, 1.00, 1.14),
            (1.14, 1.25, 1.10, 1.22),
        ]
    )
    gap = detect_fair_value_gaps(series, atr=0.1)[0]
    assert gap.side is ImbalanceSide.BULLISH
    assert gap.gap_low == pytest.approx(1.05)
    assert gap.gap_high == pytest.approx(1.10)
    assert gap.active


def test_detect_bearish_fair_value_gap(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.20, 1.25, 1.15, 1.22),
            (1.22, 1.23, 1.05, 1.07),
            (1.07, 1.10, 0.95, 0.98),
        ]
    )
    gap = detect_fair_value_gaps(series, atr=0.1)[0]
    assert gap.side is ImbalanceSide.BEARISH
    assert gap.gap_low == pytest.approx(1.10)
    assert gap.gap_high == pytest.approx(1.15)


def test_small_gap_is_filtered(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.0, 1.05, 0.95, 1.0),
            (1.0, 1.06, 0.99, 1.05),
            (1.052, 1.10, 1.051, 1.08),
        ]
    )
    assert detect_fair_value_gaps(
        series,
        atr=0.1,
        config=ImbalanceConfig(minimum_gap_atr=0.05),
    ) == ()


def test_partial_and_full_fill_tracking(make_series_ohlc: Any) -> None:
    partial = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.15, 1.00, 1.14),
            (1.14, 1.25, 1.10, 1.22),
            (1.22, 1.23, 1.075, 1.15),
        ]
    )
    gap = detect_fair_value_gaps(partial, atr=0.1)[0]
    assert gap.fill_fraction == pytest.approx(0.5)
    assert gap.filled_at is None

    full = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.15, 1.00, 1.14),
            (1.14, 1.25, 1.10, 1.22),
            (1.22, 1.23, 1.04, 1.15),
        ]
    )
    filled = detect_fair_value_gaps(full, atr=0.1)[0]
    assert filled.fill_fraction == 1.0
    assert filled.filled_at == full.bars[3].close_time


def test_nearest_fair_value_gap(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.15, 1.00, 1.14),
            (1.14, 1.25, 1.10, 1.22),
        ]
    )
    gap = detect_fair_value_gaps(series, atr=0.1)[0]
    assert nearest_fair_value_gap(
        (gap,),
        direction=StructureDirection.BULLISH,
        price=1.2,
    ) is gap
    assert nearest_fair_value_gap(
        (gap,),
        direction=StructureDirection.BEARISH,
        price=1.2,
    ) is None
