"""Channel-breakout strategy with false-breakout and liquidity-sweep rejection."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import fmean

from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.filters import (
    MarketFilter,
    StrategyFilterConfig,
    clamp_confidence,
    format_metadata,
)
from fxbot.strategy.indicators import IndicatorConfig, IndicatorError, calculate_indicators
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyConfig, StrategyDecision
from fxbot.strategy.regime import RegimeClassifier


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _default_filters() -> StrategyFilterConfig:
    return StrategyFilterConfig(
        allowed_regimes=(
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.VOLATILE,
            MarketRegime.UNKNOWN,
        ),
        max_spread_to_atr=0.25,
        max_atr_fraction=0.03,
    )


@dataclass(frozen=True, slots=True)
class BreakoutStrategyConfig:
    """Channel, confirmation, volume, and reward parameters for breakouts."""

    strategy: StrategyConfig
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    filters: StrategyFilterConfig = field(default_factory=_default_filters)
    channel_lookback: int = 20
    breakout_buffer_atr: float = 0.10
    stop_buffer_atr: float = 0.50
    target_reward_to_risk: float = 2.0
    minimum_volume_ratio: float = 0.80
    require_body_confirmation: bool = True

    def __post_init__(self) -> None:
        if self.channel_lookback < 2:
            raise ValueError("channel_lookback must be at least 2")
        object.__setattr__(
            self,
            "breakout_buffer_atr",
            _non_negative(self.breakout_buffer_atr, "breakout_buffer_atr"),
        )
        object.__setattr__(
            self,
            "stop_buffer_atr",
            _positive(self.stop_buffer_atr, "stop_buffer_atr"),
        )
        object.__setattr__(
            self,
            "target_reward_to_risk",
            _positive(self.target_reward_to_risk, "target_reward_to_risk"),
        )
        object.__setattr__(
            self,
            "minimum_volume_ratio",
            _non_negative(self.minimum_volume_ratio, "minimum_volume_ratio"),
        )


class BreakoutStrategy:
    """Trade confirmed channel closes while rejecting wick-only liquidity sweeps."""

    def __init__(
        self,
        settings: BreakoutStrategyConfig,
        *,
        regime_classifier: RegimeClassifier | None = None,
    ) -> None:
        self.settings = settings
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.market_filter = MarketFilter(settings.filters)

    @property
    def config(self) -> StrategyConfig:
        return self.settings.strategy

    def evaluate(self, context: MultiTimeframeContext) -> StrategyDecision:
        primary = context.primary
        required_bars = self.settings.channel_lookback + 1
        if len(primary.bars) < required_bars:
            return self._hold(context, "breakout_channel_warmup_incomplete")

        try:
            indicators = calculate_indicators(primary, self.settings.indicators)
            regime = self.regime_classifier.assess(primary)
        except (IndicatorError, KeyError, ValueError) as exc:
            return self._hold(
                context,
                "breakout_indicator_or_regime_unavailable",
                metadata=(("error", str(exc)),),
            )

        filter_result = self.market_filter.evaluate(
            context=context,
            indicators=indicators,
            regime=regime,
        )
        if not filter_result.allowed:
            return self._hold(
                context,
                "breakout_market_filter_rejected",
                regime=regime.regime,
                metadata=tuple(
                    (f"filter_{index}", reason)
                    for index, reason in enumerate(filter_result.reasons)
                ),
            )

        channel_bars = primary.bars[-required_bars:-1]
        latest = primary.latest
        previous = primary.bars[-2]
        upper = max(bar.mid.high for bar in channel_bars)
        lower = min(bar.mid.low for bar in channel_bars)
        atr = indicators.value("atr")
        buffer = atr * self.settings.breakout_buffer_atr
        close = latest.mid.close
        volume_average = fmean(float(bar.tick_volume) for bar in channel_bars)
        volume_ratio = (
            float(latest.tick_volume) / volume_average if volume_average > 0.0 else 1.0
        )

        bullish_sweep = latest.mid.high > upper and close <= upper
        bearish_sweep = latest.mid.low < lower and close >= lower
        if bullish_sweep:
            return self._hold(
                context,
                "bullish_liquidity_sweep_rejected",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "channel_upper": upper,
                        "close": close,
                        "latest_high": latest.mid.high,
                    }
                ),
            )
        if bearish_sweep:
            return self._hold(
                context,
                "bearish_liquidity_sweep_rejected",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "channel_lower": lower,
                        "close": close,
                        "latest_low": latest.mid.low,
                    }
                ),
            )

        bullish_body = latest.mid.close > latest.mid.open
        bearish_body = latest.mid.close < latest.mid.open
        long_candidate = (
            close > upper + buffer
            and previous.mid.close <= upper
            and volume_ratio >= self.settings.minimum_volume_ratio
            and (bullish_body or not self.settings.require_body_confirmation)
            and regime.regime is not MarketRegime.TRENDING_DOWN
        )
        short_candidate = (
            close < lower - buffer
            and previous.mid.close >= lower
            and volume_ratio >= self.settings.minimum_volume_ratio
            and (bearish_body or not self.settings.require_body_confirmation)
            and regime.regime is not MarketRegime.TRENDING_UP
        )

        action: SignalAction | None = None
        if long_candidate:
            action = SignalAction.BUY
        elif short_candidate:
            action = SignalAction.SELL
        if action is None:
            return self._hold(
                context,
                "breakout_close_confirmation_missing",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "breakout_buffer": buffer,
                        "channel_lower": lower,
                        "channel_upper": upper,
                        "close": close,
                        "volume_ratio": volume_ratio,
                    }
                ),
            )

        if action is SignalAction.BUY:
            stop = upper - atr * self.settings.stop_buffer_atr
            risk = close - stop
            target = close + risk * self.settings.target_reward_to_risk
            breakout_distance = close - upper
        else:
            stop = lower + atr * self.settings.stop_buffer_atr
            risk = stop - close
            target = close - risk * self.settings.target_reward_to_risk
            breakout_distance = lower - close
        if stop <= 0.0 or target <= 0.0 or risk <= 0.0:
            return self._hold(
                context,
                "breakout_invalid_price_bracket",
                regime=regime.regime,
            )

        distance_score = min(1.0, breakout_distance / max(atr, 1e-12))
        volume_score = min(1.0, volume_ratio / max(self.settings.minimum_volume_ratio * 2.0, 1e-12))
        close_location = _close_location(latest.mid.high, latest.mid.low, close, action)
        confidence = clamp_confidence(
            0.35 * distance_score
            + 0.25 * volume_score
            + 0.20 * close_location
            + 0.20 * regime.confidence
        )
        confidence *= filter_result.confidence_multiplier

        return StrategyDecision(
            strategy_id=self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            action=action,
            as_of=context.as_of,
            confidence=clamp_confidence(confidence),
            reasons=(
                "channel_boundary_broken",
                "close_confirmed_outside_channel",
                "volume_confirmation_passed",
                "liquidity_sweep_rejection_passed",
            ),
            regime=regime.regime,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            metadata=format_metadata(
                {
                    "atr": atr,
                    "breakout_distance_atr": breakout_distance / max(atr, 1e-12),
                    "channel_lower": lower,
                    "channel_upper": upper,
                    "reward_to_risk": self.settings.target_reward_to_risk,
                    "volume_ratio": volume_ratio,
                }
            ),
        )

    def _hold(
        self,
        context: MultiTimeframeContext,
        reason: str,
        *,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> StrategyDecision:
        return StrategyDecision.hold(
            strategy_id=self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            as_of=context.as_of,
            reason=reason,
            regime=regime,
            metadata=metadata,
        )


def _close_location(
    high: float,
    low: float,
    close: float,
    action: SignalAction,
) -> float:
    width = high - low
    if width <= 0.0:
        return 0.5
    fraction = (close - low) / width
    return fraction if action is SignalAction.BUY else 1.0 - fraction
