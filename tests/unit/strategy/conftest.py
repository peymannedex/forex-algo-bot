from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.strategy.context import MarketSeries, MultiTimeframeContext
from fxbot.strategy.models import (
    IndicatorSnapshot,
    MarketRegime,
    RegimeAssessment,
    RegimeConfluence,
)

BASE = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def _bar(
    *,
    index: int,
    close: float,
    previous_close: float,
    timeframe: Timeframe,
    spread: float,
    tick_volume: int,
    open_override: float | None = None,
    high_override: float | None = None,
    low_override: float | None = None,
) -> Bar:
    open_price = previous_close if open_override is None else open_override
    high = max(open_price, close) + 0.0005 if high_override is None else high_override
    low = min(open_price, close) - 0.0005 if low_override is None else low_override
    mid = OHLC(open_price, high, low, close)
    half = spread / 2.0
    bid = OHLC(open_price - half, high - half, low - half, close - half)
    ask = OHLC(open_price + half, high + half, low + half, close + half)
    return Bar(
        symbol="EURUSD",
        open_time=BASE + timedelta(seconds=(timeframe.seconds or 60) * index),
        timeframe=timeframe,
        bid=bid,
        ask=ask,
        mid_ohlc=mid,
        tick_volume=tick_volume,
    )


@pytest.fixture
def make_context() -> Any:
    def factory(
        closes: list[float],
        *,
        timeframe: Timeframe = Timeframe.M5,
        higher_closes: list[float] | None = None,
        latest_open: float | None = None,
        latest_high: float | None = None,
        latest_low: float | None = None,
        latest_volume: int = 100,
        spread: float = 0.0001,
    ) -> MultiTimeframeContext:
        bars: list[Bar] = []
        for index, close in enumerate(closes):
            previous = closes[index - 1] if index else close
            is_latest = index == len(closes) - 1
            bars.append(
                _bar(
                    index=index,
                    close=close,
                    previous_close=previous,
                    timeframe=timeframe,
                    spread=spread,
                    tick_volume=latest_volume if is_latest else 100,
                    open_override=latest_open if is_latest else None,
                    high_override=latest_high if is_latest else None,
                    low_override=latest_low if is_latest else None,
                )
            )
        series = [MarketSeries("EURUSD", timeframe, tuple(bars))]
        if higher_closes is not None:
            higher_bars = tuple(
                _bar(
                    index=index,
                    close=close,
                    previous_close=higher_closes[index - 1] if index else close,
                    timeframe=Timeframe.H1,
                    spread=spread,
                    tick_volume=100,
                )
                for index, close in enumerate(higher_closes)
            )
            series.append(MarketSeries("EURUSD", Timeframe.H1, higher_bars))
        as_of = max(item.latest.close_time for item in series)
        return MultiTimeframeContext("EURUSD", as_of, timeframe, tuple(series))

    return factory


@pytest.fixture
def make_snapshot() -> Any:
    def factory(
        context: MultiTimeframeContext,
        *,
        atr: float = 0.002,
        fast: float = 1.101,
        slow: float = 1.100,
        momentum: float = 0.002,
        rsi: float = 60.0,
        spread_to_atr: float = 0.05,
        atr_fraction: float = 0.002,
    ) -> IndicatorSnapshot:
        return IndicatorSnapshot(
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            as_of=context.primary.latest.close_time,
            values=(
                ("atr", atr),
                ("atr_fraction", atr_fraction),
                ("average_spread", atr * spread_to_atr),
                ("fast_ema", fast),
                ("momentum", momentum),
                ("realized_volatility", 0.001),
                ("rsi", rsi),
                ("slow_ema", slow),
                ("sma", (fast + slow) / 2.0),
                ("spread_to_atr", spread_to_atr),
            ),
            sample_size=len(context.primary.bars),
        )

    return factory


class StubRegimeClassifier:
    def __init__(
        self,
        regime: MarketRegime,
        *,
        dominant: MarketRegime | None = None,
        alignment: float = 1.0,
        confidence: float = 0.8,
    ) -> None:
        self.regime = regime
        self.dominant = dominant or regime
        self.alignment = alignment
        self.confidence = confidence

    def assess(self, series: MarketSeries) -> RegimeAssessment:
        return RegimeAssessment(
            symbol=series.symbol,
            timeframe=series.timeframe,
            as_of=series.latest.close_time,
            regime=self.regime,
            confidence=self.confidence,
            metrics=(),
            reasons=("test regime",),
        )

    def confluence(
        self,
        context: MultiTimeframeContext,
        timeframes: tuple[Timeframe, ...] | None = None,
    ) -> RegimeConfluence:
        selected = context.timeframes if timeframes is None else timeframes
        return RegimeConfluence(
            symbol=context.symbol,
            as_of=context.as_of,
            primary_timeframe=context.primary_timeframe,
            primary_regime=self.regime,
            dominant_regime=self.dominant,
            alignment_score=self.alignment,
            assessments=tuple((timeframe, self.dominant) for timeframe in selected),
        )


@pytest.fixture
def classifier_factory() -> Any:
    def factory(
        regime: MarketRegime,
        *,
        dominant: MarketRegime | None = None,
        alignment: float = 1.0,
        confidence: float = 0.8,
    ) -> StubRegimeClassifier:
        return StubRegimeClassifier(
            regime,
            dominant=dominant,
            alignment=alignment,
            confidence=confidence,
        )

    return factory

@pytest.fixture
def make_series_ohlc() -> Any:
    def factory(
        rows: list[tuple[float, float, float, float]],
        *,
        timeframe: Timeframe = Timeframe.M5,
        spread: float = 0.0001,
        volumes: list[int] | None = None,
    ) -> MarketSeries:
        bars: list[Bar] = []
        for index, (open_price, high, low, close) in enumerate(rows):
            mid = OHLC(open_price, high, low, close)
            half = spread / 2.0
            bid = OHLC(open_price - half, high - half, low - half, close - half)
            ask = OHLC(open_price + half, high + half, low + half, close + half)
            bars.append(
                Bar(
                    symbol="EURUSD",
                    open_time=BASE + timedelta(seconds=(timeframe.seconds or 60) * index),
                    timeframe=timeframe,
                    bid=bid,
                    ask=ask,
                    mid_ohlc=mid,
                    tick_volume=100 if volumes is None else volumes[index],
                )
            )
        return MarketSeries("EURUSD", timeframe, tuple(bars))

    return factory
