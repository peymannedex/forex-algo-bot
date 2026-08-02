"""Deterministic market-regime classification and timeframe confluence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MarketSeries, MultiTimeframeContext
from fxbot.strategy.indicators import IndicatorConfig, calculate_indicators
from fxbot.strategy.models import (
    MarketRegime,
    RegimeAssessment,
    RegimeConfluence,
)


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    illiquid_spread_to_atr: float = 0.35
    high_volatility_atr_fraction: float = 0.004
    trend_separation_atr: float = 0.20
    trend_momentum_threshold: float = 0.001
    range_separation_atr: float = 0.10
    range_momentum_threshold: float = 0.0008

    def __post_init__(self) -> None:
        for name in (
            "illiquid_spread_to_atr",
            "high_volatility_atr_fraction",
            "trend_separation_atr",
            "trend_momentum_threshold",
            "range_separation_atr",
            "range_momentum_threshold",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.range_separation_atr > self.trend_separation_atr:
            raise ValueError("range_separation_atr cannot exceed trend_separation_atr")


class RegimeClassifier:
    """Classify liquidity, volatility, trend, and range states in priority order."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def assess(self, series: MarketSeries) -> RegimeAssessment:
        snapshot = calculate_indicators(series, self.config.indicators)
        atr = snapshot.value("atr")
        fast = snapshot.value("fast_ema")
        slow = snapshot.value("slow_ema")
        momentum = snapshot.value("momentum")
        spread_ratio = snapshot.value("spread_to_atr")
        atr_fraction = snapshot.value("atr_fraction")
        latest_close = series.latest.mid.close
        separation = abs(fast - slow) / atr if atr > 0.0 else 0.0

        metrics = tuple(
            sorted(
                (
                    ("atr_fraction", atr_fraction),
                    ("ema_separation_atr", separation),
                    ("fast_ema", fast),
                    ("latest_close", latest_close),
                    ("momentum", momentum),
                    ("slow_ema", slow),
                    ("spread_to_atr", spread_ratio),
                )
            )
        )

        if spread_ratio >= self.config.illiquid_spread_to_atr:
            return RegimeAssessment(
                symbol=series.symbol,
                timeframe=series.timeframe,
                as_of=snapshot.as_of,
                regime=MarketRegime.ILLIQUID,
                confidence=_bounded_ratio(spread_ratio, self.config.illiquid_spread_to_atr),
                metrics=metrics,
                reasons=("Spread is large relative to ATR",),
            )

        if atr_fraction >= self.config.high_volatility_atr_fraction:
            return RegimeAssessment(
                symbol=series.symbol,
                timeframe=series.timeframe,
                as_of=snapshot.as_of,
                regime=MarketRegime.VOLATILE,
                confidence=_bounded_ratio(
                    atr_fraction,
                    self.config.high_volatility_atr_fraction,
                ),
                metrics=metrics,
                reasons=("ATR fraction exceeds high-volatility threshold",),
            )

        if (
            separation >= self.config.trend_separation_atr
            and fast > slow
            and latest_close > slow
            and momentum >= self.config.trend_momentum_threshold
        ):
            return RegimeAssessment(
                symbol=series.symbol,
                timeframe=series.timeframe,
                as_of=snapshot.as_of,
                regime=MarketRegime.TRENDING_UP,
                confidence=_trend_confidence(separation, momentum, self.config),
                metrics=metrics,
                reasons=("Fast EMA is above slow EMA with positive momentum",),
            )

        if (
            separation >= self.config.trend_separation_atr
            and fast < slow
            and latest_close < slow
            and momentum <= -self.config.trend_momentum_threshold
        ):
            return RegimeAssessment(
                symbol=series.symbol,
                timeframe=series.timeframe,
                as_of=snapshot.as_of,
                regime=MarketRegime.TRENDING_DOWN,
                confidence=_trend_confidence(separation, abs(momentum), self.config),
                metrics=metrics,
                reasons=("Fast EMA is below slow EMA with negative momentum",),
            )

        if (
            separation <= self.config.range_separation_atr
            and abs(momentum) <= self.config.range_momentum_threshold
        ):
            confidence = 1.0 - min(
                1.0,
                max(
                    separation / max(self.config.range_separation_atr, 1e-12),
                    abs(momentum) / max(self.config.range_momentum_threshold, 1e-12),
                ),
            )
            return RegimeAssessment(
                symbol=series.symbol,
                timeframe=series.timeframe,
                as_of=snapshot.as_of,
                regime=MarketRegime.RANGING,
                confidence=max(confidence, 0.5),
                metrics=metrics,
                reasons=("EMA separation and momentum are both compressed",),
            )

        return RegimeAssessment(
            symbol=series.symbol,
            timeframe=series.timeframe,
            as_of=snapshot.as_of,
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            metrics=metrics,
            reasons=("No configured regime threshold was satisfied",),
        )

    def confluence(
        self,
        context: MultiTimeframeContext,
        timeframes: tuple[Timeframe, ...] | None = None,
    ) -> RegimeConfluence:
        selected = context.timeframes if timeframes is None else timeframes
        if not selected:
            raise ValueError("At least one timeframe is required")
        assessments = tuple(self.assess(context.get(timeframe)) for timeframe in selected)
        counts = Counter(item.regime for item in assessments)
        dominant, count = counts.most_common(1)[0]
        primary = next(
            item for item in assessments if item.timeframe is context.primary_timeframe
        )
        return RegimeConfluence(
            symbol=context.symbol,
            as_of=context.as_of,
            primary_timeframe=context.primary_timeframe,
            primary_regime=primary.regime,
            dominant_regime=dominant,
            alignment_score=count / len(assessments),
            assessments=tuple((item.timeframe, item.regime) for item in assessments),
        )


def _bounded_ratio(observed: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 1.0
    return min(1.0, 0.5 + 0.5 * (observed - threshold) / threshold)


def _trend_confidence(
    separation: float,
    momentum_value: float,
    config: RegimeConfig,
) -> float:
    separation_score = min(
        1.0,
        separation / max(config.trend_separation_atr * 2.0, 1e-12),
    )
    momentum_score = min(
        1.0,
        momentum_value / max(config.trend_momentum_threshold * 3.0, 1e-12),
    )
    return max(0.5, (separation_score + momentum_score) / 2.0)
