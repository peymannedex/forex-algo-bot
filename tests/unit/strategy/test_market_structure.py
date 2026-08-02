from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.market_structure import (
    DealingRangeZone,
    MarketStructureConfig,
    StructureDirection,
    StructureEventKind,
    SwingKind,
    SwingPoint,
    analyze_market_structure,
    detect_displacements,
    detect_structure_events,
    detect_swings,
    structure_confluence,
)


def test_market_structure_config_validation() -> None:
    with pytest.raises(ValueError):
        MarketStructureConfig(left_bars=0)
    with pytest.raises(ValueError):
        MarketStructureConfig(displacement_close_fraction=0.4)
    with pytest.raises(ValueError):
        MarketStructureConfig(dealing_range_lookback_swings=1)


def test_detect_swings_is_confirmed_after_right_bars(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.01),
            (1.01, 1.10, 0.98, 1.05),
            (1.05, 1.20, 1.00, 1.10),
            (1.10, 1.15, 0.99, 1.02),
            (1.02, 1.08, 0.96, 1.00),
        ]
    )

    swings = detect_swings(series, left_bars=2, right_bars=2)

    high = next(item for item in swings if item.kind is SwingKind.HIGH)
    assert high.index == 2
    assert high.price == pytest.approx(1.20)
    assert high.confirmed_at == series.bars[4].close_time


def test_detect_swings_returns_empty_when_warmup_missing(make_series_ohlc: Any) -> None:
    series = make_series_ohlc([(1.0, 1.1, 0.9, 1.0)] * 4)
    assert detect_swings(series, left_bars=2, right_bars=2) == ()


def test_detect_structure_events_classifies_bos_then_choch(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.05, 1.10, 1.00, 1.05),
            (1.05, 1.20, 1.02, 1.10),
            (1.10, 1.15, 1.04, 1.08),
            (1.08, 1.25, 1.06, 1.22),
            (1.22, 1.23, 1.00, 1.08),
            (1.08, 1.15, 1.03, 1.10),
            (1.10, 1.12, 0.95, 0.98),
        ]
    )
    high = SwingPoint(
        "EURUSD",
        Timeframe.M5,
        1,
        SwingKind.HIGH,
        1.20,
        series.bars[1].open_time,
        series.bars[2].close_time,
    )
    low = SwingPoint(
        "EURUSD",
        Timeframe.M5,
        4,
        SwingKind.LOW,
        1.00,
        series.bars[4].open_time,
        series.bars[5].close_time,
    )

    events = detect_structure_events(series, (high, low))

    assert [item.kind for item in events] == [
        StructureEventKind.BREAK_OF_STRUCTURE,
        StructureEventKind.CHANGE_OF_CHARACTER,
    ]
    assert [item.direction for item in events] == [
        StructureDirection.BULLISH,
        StructureDirection.BEARISH,
    ]


def test_structure_break_requires_close_beyond_buffer(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.0, 1.1, 0.9, 1.0),
            (1.0, 1.2, 0.95, 1.1),
            (1.1, 1.18, 1.0, 1.08),
            (1.08, 1.23, 1.05, 1.205),
        ]
    )
    swing = SwingPoint(
        "EURUSD",
        Timeframe.M5,
        1,
        SwingKind.HIGH,
        1.20,
        series.bars[1].open_time,
        series.bars[2].close_time,
    )
    assert detect_structure_events(series, (swing,), break_buffer=0.01) == ()


def test_detect_displacements_bullish_and_bearish(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.03, 0.99, 1.02),
            (1.02, 1.16, 1.01, 1.15),
            (1.15, 1.16, 1.00, 1.01),
        ]
    )

    result = detect_displacements(
        series,
        atr=0.10,
        body_atr_threshold=1.0,
        close_fraction=0.7,
    )

    assert [item.direction for item in result] == [
        StructureDirection.BULLISH,
        StructureDirection.BEARISH,
    ]
    assert result[0].body_atr == pytest.approx(1.3)


def test_analyze_market_structure_builds_dealing_range(make_series_ohlc: Any) -> None:
    series = make_series_ohlc(
        [
            (1.00, 1.05, 0.95, 1.00),
            (1.00, 1.20, 0.98, 1.10),
            (1.10, 1.15, 1.00, 1.05),
            (1.05, 1.10, 0.80, 0.90),
            (0.90, 1.00, 0.85, 0.95),
            (0.95, 1.10, 0.90, 1.08),
        ]
    )
    state = analyze_market_structure(
        series,
        atr=0.10,
        config=MarketStructureConfig(left_bars=1, right_bars=1),
    )

    assert state.range_high == pytest.approx(1.20)
    assert state.range_low == pytest.approx(0.80)
    assert state.equilibrium == pytest.approx(1.00)
    assert state.price_zone is DealingRangeZone.PREMIUM


def test_structure_confluence_requires_atr_for_each_timeframe(make_series_ohlc: Any) -> None:
    primary = make_series_ohlc([(1.0, 1.1, 0.9, 1.0)] * 6)
    higher = make_series_ohlc(
        [(1.0, 1.1, 0.9, 1.0)] * 6,
        timeframe=Timeframe.H1,
    )
    context = MultiTimeframeContext(
        "EURUSD",
        max(primary.latest.close_time, higher.latest.close_time),
        Timeframe.M5,
        (primary, higher),
    )

    with pytest.raises(KeyError, match="Missing ATR"):
        structure_confluence(context, atr_by_timeframe={Timeframe.M5: 0.1})


def test_swing_point_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SwingPoint(
            "EURUSD",
            Timeframe.M5,
            1,
            SwingKind.HIGH,
            1.2,
            datetime(2026, 1, 1),
            datetime(2026, 1, 1) + timedelta(minutes=5),
        )


def test_structure_confluence_returns_neutral_alignment(make_series_ohlc: Any) -> None:
    primary = make_series_ohlc([(1.0, 1.1, 0.9, 1.0)] * 6)
    higher = make_series_ohlc(
        [(1.0, 1.1, 0.9, 1.0)] * 6,
        timeframe=Timeframe.H1,
    )
    context = MultiTimeframeContext(
        "EURUSD",
        max(primary.latest.close_time, higher.latest.close_time),
        Timeframe.M5,
        (primary, higher),
    )

    result = structure_confluence(
        context,
        atr_by_timeframe={Timeframe.M5: 0.1, Timeframe.H1: 0.1},
        config=MarketStructureConfig(left_bars=1, right_bars=1),
    )

    assert result.dominant_bias is StructureDirection.NEUTRAL
    assert result.alignment_score == 1.0
    assert result.as_of.tzinfo is UTC
