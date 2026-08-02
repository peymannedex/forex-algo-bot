from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.strategy.context import MarketSeries, MultiTimeframeContext
from fxbot.strategy.indicators import IndicatorConfig
from fxbot.strategy.models import MarketRegime
from fxbot.strategy.regime import RegimeClassifier, RegimeConfig

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_series(
    prices: list[float],
    *,
    timeframe: Timeframe = Timeframe.M5,
    spread: float = 0.00005,
    range_size: float = 0.0002,
) -> MarketSeries:
    seconds = timeframe.seconds
    assert seconds is not None
    bars: list[Bar] = []
    for index, close in enumerate(prices):
        bid_close = close - spread
        bars.append(
            Bar(
                symbol="EURUSD",
                open_time=BASE + timedelta(seconds=index * seconds),
                timeframe=timeframe,
                bid=OHLC(
                    bid_close,
                    bid_close + range_size,
                    bid_close - range_size,
                    bid_close,
                ),
                ask=OHLC(
                    close,
                    close + range_size,
                    close - range_size,
                    close,
                ),
            )
        )
    return MarketSeries("EURUSD", timeframe, tuple(bars))


def classifier(**overrides: float) -> RegimeClassifier:
    indicators = IndicatorConfig(
        fast_ema_period=3,
        slow_ema_period=6,
        sma_period=5,
        atr_period=3,
        rsi_period=3,
        momentum_period=3,
        volatility_period=5,
        spread_period=5,
    )
    config = RegimeConfig(indicators=indicators, **overrides)
    return RegimeClassifier(config)


def test_classifies_uptrend() -> None:
    series = make_series([1.10 + i * 0.0005 for i in range(20)])
    result = classifier(high_volatility_atr_fraction=1.0).assess(series)
    assert result.regime is MarketRegime.TRENDING_UP
    assert result.confidence >= 0.5


def test_classifies_downtrend() -> None:
    series = make_series([1.12 - i * 0.0005 for i in range(20)])
    result = classifier(high_volatility_atr_fraction=1.0).assess(series)
    assert result.regime is MarketRegime.TRENDING_DOWN


def test_classifies_range() -> None:
    series = make_series([1.10] * 20)
    result = classifier(high_volatility_atr_fraction=1.0).assess(series)
    assert result.regime is MarketRegime.RANGING


def test_illiquidity_has_priority() -> None:
    series = make_series([1.10 + i * 0.0001 for i in range(20)], spread=0.001)
    result = classifier(illiquid_spread_to_atr=0.2).assess(series)
    assert result.regime is MarketRegime.ILLIQUID


def test_high_volatility_precedes_trend() -> None:
    prices = [1.10 + (0.01 if i % 2 else -0.01) for i in range(20)]
    series = make_series(prices, range_size=0.003)
    result = classifier(
        illiquid_spread_to_atr=1.0,
        high_volatility_atr_fraction=0.001,
    ).assess(series)
    assert result.regime is MarketRegime.VOLATILE


def test_unknown_when_between_thresholds() -> None:
    prices = [1.10 + i * 0.00005 for i in range(20)]
    result = classifier(
        high_volatility_atr_fraction=1.0,
        trend_separation_atr=10.0,
        range_separation_atr=0.0,
        range_momentum_threshold=0.0,
    ).assess(make_series(prices))
    assert result.regime is MarketRegime.UNKNOWN


def test_multi_timeframe_confluence() -> None:
    m5 = make_series([1.10 + i * 0.0005 for i in range(20)], timeframe=Timeframe.M5)
    h1 = make_series([1.10 + i * 0.0005 for i in range(20)], timeframe=Timeframe.H1)
    context = MultiTimeframeContext(
        symbol="EURUSD",
        as_of=h1.latest.close_time,
        primary_timeframe=Timeframe.M5,
        series=(m5, h1),
    )
    result = classifier(high_volatility_atr_fraction=1.0).confluence(context)
    assert result.dominant_regime is MarketRegime.TRENDING_UP
    assert result.alignment_score == 1.0
    assert result.directionally_aligned
