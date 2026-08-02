"""Look-ahead-safe technical indicators and incremental rolling state."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, log, sqrt
from statistics import fmean, pstdev

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import Bar
from fxbot.strategy.context import MarketSeries
from fxbot.strategy.models import IndicatorSnapshot


class IndicatorError(ValueError):
    """Raised when an indicator cannot be calculated from supplied observations."""


def _period(value: int, field_name: str = "period") -> int:
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _finite_values(values: Sequence[float], minimum: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) < minimum:
        raise IndicatorError(f"{name} requires at least {minimum} observations")
    if not all(isfinite(value) for value in result):
        raise IndicatorError(f"{name} values must be finite")
    return result


def simple_moving_average(values: Sequence[float], period: int) -> float:
    period = _period(period)
    samples = _finite_values(values, period, "SMA")
    return fmean(samples[-period:])


def exponential_moving_average(values: Sequence[float], period: int) -> float:
    """EMA seeded with an SMA, then updated only with subsequent observations."""

    period = _period(period)
    samples = _finite_values(values, period, "EMA")
    current = fmean(samples[:period])
    multiplier = 2.0 / (period + 1.0)
    for value in samples[period:]:
        current += multiplier * (value - current)
    return current


def true_range(current: Bar, previous_close: float) -> float:
    if not isfinite(previous_close) or previous_close <= 0.0:
        raise ValueError("previous_close must be a positive finite number")
    high = current.mid.high
    low = current.mid.low
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def average_true_range(bars: Sequence[Bar], period: int) -> float:
    """Wilder ATR based on completed observations in chronological order."""

    period = _period(period)
    if len(bars) < period + 1:
        raise IndicatorError(f"ATR requires at least {period + 1} bars")
    ranges = [true_range(current, previous.mid.close) for previous, current in pairwise(bars)]
    current_atr = fmean(ranges[:period])
    for value in ranges[period:]:
        current_atr = ((period - 1) * current_atr + value) / period
    return current_atr


def relative_strength_index(values: Sequence[float], period: int) -> float:
    """Wilder RSI with stable handling of one-sided and flat markets."""

    period = _period(period)
    samples = _finite_values(values, period + 1, "RSI")
    changes = [current - previous for previous, current in pairwise(samples)]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    average_gain = fmean(gains[:period])
    average_loss = fmean(losses[:period])
    for gain, loss in zip(gains[period:], losses[period:], strict=False):
        average_gain = ((period - 1) * average_gain + gain) / period
        average_loss = ((period - 1) * average_loss + loss) / period
    if average_loss == 0.0:
        return 50.0 if average_gain == 0.0 else 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def momentum(values: Sequence[float], period: int) -> float:
    """Fractional price change over ``period`` observations."""

    period = _period(period)
    samples = _finite_values(values, period + 1, "Momentum")
    initial = samples[-period - 1]
    if initial <= 0.0:
        raise IndicatorError("Momentum requires positive prices")
    return samples[-1] / initial - 1.0


def realized_volatility(
    values: Sequence[float],
    period: int,
    *,
    annualization: float = 1.0,
) -> float:
    """Population standard deviation of log returns with optional scaling."""

    period = _period(period)
    if not isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization must be a positive finite number")
    samples = _finite_values(values, period + 1, "Realized volatility")
    selected = samples[-period - 1 :]
    if any(value <= 0.0 for value in selected):
        raise IndicatorError("Realized volatility requires positive prices")
    returns = [log(current / previous) for previous, current in pairwise(selected)]
    return pstdev(returns) * sqrt(annualization)


def average_spread(bars: Sequence[Bar], period: int) -> float:
    period = _period(period)
    if len(bars) < period:
        raise IndicatorError(f"Average spread requires at least {period} bars")
    return fmean(bar.spread_close for bar in bars[-period:])


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    fast_ema_period: int = 12
    slow_ema_period: int = 26
    sma_period: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    momentum_period: int = 10
    volatility_period: int = 20
    spread_period: int = 20
    annualization: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "fast_ema_period",
            "slow_ema_period",
            "sma_period",
            "atr_period",
            "rsi_period",
            "momentum_period",
            "volatility_period",
            "spread_period",
        ):
            _period(int(getattr(self, name)), name)
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("fast_ema_period must be below slow_ema_period")
        if not isfinite(self.annualization) or self.annualization <= 0.0:
            raise ValueError("annualization must be a positive finite number")

    @property
    def minimum_bars(self) -> int:
        return max(
            self.slow_ema_period,
            self.sma_period,
            self.atr_period + 1,
            self.rsi_period + 1,
            self.momentum_period + 1,
            self.volatility_period + 1,
            self.spread_period,
        )


def calculate_indicators(
    series: MarketSeries,
    config: IndicatorConfig | None = None,
) -> IndicatorSnapshot:
    """Calculate a canonical indicator snapshot from the series tail."""

    settings = config or IndicatorConfig()
    if len(series.bars) < settings.minimum_bars:
        raise IndicatorError(
            f"Indicator set requires {settings.minimum_bars} bars; got {len(series.bars)}"
        )
    closes = series.closes
    atr = average_true_range(series.bars, settings.atr_period)
    spread = average_spread(series.bars, settings.spread_period)
    latest_close = closes[-1]
    values = (
        ("atr", atr),
        ("average_spread", spread),
        ("fast_ema", exponential_moving_average(closes, settings.fast_ema_period)),
        ("momentum", momentum(closes, settings.momentum_period)),
        (
            "realized_volatility",
            realized_volatility(
                closes,
                settings.volatility_period,
                annualization=settings.annualization,
            ),
        ),
        ("rsi", relative_strength_index(closes, settings.rsi_period)),
        ("slow_ema", exponential_moving_average(closes, settings.slow_ema_period)),
        ("sma", simple_moving_average(closes, settings.sma_period)),
        ("spread_to_atr", spread / atr if atr > 0.0 else 0.0),
        ("atr_fraction", atr / latest_close),
    )
    return IndicatorSnapshot(
        symbol=series.symbol,
        timeframe=series.timeframe,
        as_of=series.latest.close_time,
        values=values,
        sample_size=len(series.bars),
    )


class RollingIndicatorState:
    """Bounded incremental bar state that rejects out-of-order observations."""

    def __init__(self, symbol: str, timeframe: Timeframe, *, max_bars: int = 500) -> None:
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol cannot be empty")
        parsed = Timeframe.parse(timeframe)
        if parsed is Timeframe.TICK:
            raise ValueError("RollingIndicatorState timeframe cannot be tick")
        if max_bars < 2:
            raise ValueError("max_bars must be at least 2")
        self.symbol = normalized
        self.timeframe = parsed
        self.max_bars = max_bars
        self._bars: deque[Bar] = deque(maxlen=max_bars)

    @property
    def bars(self) -> tuple[Bar, ...]:
        return tuple(self._bars)

    def append(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            raise ValueError(f"Bar symbol {bar.symbol} does not match {self.symbol}")
        if bar.timeframe is not self.timeframe:
            raise ValueError(
                f"Bar timeframe {bar.timeframe.value} does not match {self.timeframe.value}"
            )
        if self._bars and bar.open_time <= self._bars[-1].open_time:
            raise ValueError("Bars must be appended in strictly increasing order")
        self._bars.append(bar)

    def extend(self, bars: Iterable[Bar]) -> None:
        for bar in bars:
            self.append(bar)

    def snapshot(self, config: IndicatorConfig | None = None) -> IndicatorSnapshot:
        return calculate_indicators(
            MarketSeries(self.symbol, self.timeframe, tuple(self._bars)),
            config,
        )
