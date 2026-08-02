from __future__ import annotations

from typing import Any

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.market_structure import (
    StructureDirection,
    StructureEvent,
    StructureEventKind,
    SwingKind,
    SwingPoint,
)
from fxbot.strategy.order_blocks import (
    OrderBlockConfig,
    OrderBlockSide,
    detect_order_blocks,
    nearest_order_block,
)


def _event(series: Any, index: int, direction: StructureDirection) -> StructureEvent:
    kind = SwingKind.HIGH if direction is StructureDirection.BULLISH else SwingKind.LOW
    swing = SwingPoint(
        "EURUSD",
        Timeframe.M5,
        1,
        kind,
        1.1 if kind is SwingKind.HIGH else 0.9,
        series.bars[1].open_time,
        series.bars[2].close_time,
    )
    return StructureEvent(
        "EURUSD",
        Timeframe.M5,
        index,
        StructureEventKind.BREAK_OF_STRUCTURE,
        direction,
        swing.price,
        series.bars[index].mid.close,
        series.bars[index].close_time,
        swing,
    )


def test_order_block_config_validation() -> None:
    with pytest.raises(ValueError):
        OrderBlockConfig(search_lookback=0)
    with pytest.raises(ValueError):
        OrderBlockConfig(invalidation_buffer_atr=-1)


def test_detect_bullish_order_block_from_last_bearish_candle(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.06, 0.96, 0.98),
            (0.98, 1.04, 0.97, 1.01),
            (1.01, 1.25, 1.00, 1.22),
        ]
    )

    block = detect_order_blocks(series, (_event(series, 3, StructureDirection.BULLISH),), atr=0.1)[0]

    assert block.side is OrderBlockSide.BULLISH
    assert block.origin_index == 1
    assert block.zone_low == pytest.approx(0.96)
    assert block.zone_high == pytest.approx(1.06)


def test_body_zone_option(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.06, 0.96, 0.98),
            (0.98, 1.04, 0.97, 1.01),
            (1.01, 1.25, 1.00, 1.22),
        ]
    )
    block = detect_order_blocks(
        series,
        (_event(series, 3, StructureDirection.BULLISH),),
        atr=0.1,
        config=OrderBlockConfig(use_candle_body=True),
    )[0]
    assert block.zone_low == pytest.approx(0.98)
    assert block.zone_high == pytest.approx(1.02)


def test_small_break_body_is_rejected(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.06, 0.96, 0.98),
            (0.98, 1.04, 0.97, 1.01),
            (1.01, 1.25, 1.00, 1.03),
        ]
    )
    assert detect_order_blocks(
        series,
        (_event(series, 3, StructureDirection.BULLISH),),
        atr=0.1,
        config=OrderBlockConfig(minimum_break_body_atr=0.8),
    ) == ()


def test_order_block_tracks_mitigation_and_invalidation(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.06, 0.96, 0.98),
            (0.98, 1.04, 0.97, 1.01),
            (1.01, 1.25, 1.00, 1.22),
            (1.22, 1.24, 1.03, 1.15),
            (1.15, 1.16, 0.90, 0.92),
        ]
    )
    block = detect_order_blocks(series, (_event(series, 3, StructureDirection.BULLISH),), atr=0.1)[0]
    assert block.mitigated_at == series.bars[4].close_time
    assert block.invalidated_at == series.bars[5].close_time
    assert not block.active


def test_nearest_order_block(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.02),
            (1.02, 1.06, 0.96, 0.98),
            (0.98, 1.04, 0.97, 1.01),
            (1.01, 1.25, 1.00, 1.22),
        ]
    )
    block = detect_order_blocks(series, (_event(series, 3, StructureDirection.BULLISH),), atr=0.1)[0]
    assert nearest_order_block(
        (block,),
        direction=StructureDirection.BULLISH,
        price=1.1,
    ) is block
    assert nearest_order_block(
        (block,),
        direction=StructureDirection.BEARISH,
        price=1.1,
    ) is None
