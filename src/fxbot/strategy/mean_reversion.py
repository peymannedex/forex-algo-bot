"""Range-regime mean-reversion strategy with band-touch rejection entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import fmean, pstdev

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


def _rsi(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 100.0:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return number


def _default_filters() -> StrategyFilterConfig:
    return StrategyFilterConfig(
        allowed_regimes=(MarketRegime.RANGING,),
        max_spread_to_atr=0.20,
        max_atr_fraction=0.01,
    )


@dataclass(frozen=True, slots=True)
class MeanReversionConfig:
    """Parameters for band extremes, RSI exhaustion, and mean targets."""

    strategy: StrategyConfig
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    filters: StrategyFilterConfig = field(default_factory=_default_filters)
    band_period: int = 20
    band_standard_deviations: float = 2.0
    oversold_rsi: float = 35.0
    overbought_rsi: float = 65.0
    stop_atr_multiple: float = 1.0
    minimum_reward_to_risk: float = 0.75

    def __post_init__(self) -> None:
        if self.band_period < 2:
            raise ValueError("band_period must be at least 2")
        object.__setattr__(
            self,
            "band_standard_deviations",
            _positive(
                self.band_standard_deviations,
                "band_standard_deviations",
            ),
        )
        object.__setattr__(self, "oversold_rsi", _rsi(self.oversold_rsi, "oversold_rsi"))
        object.__setattr__(
            self,
            "overbought_rsi",
            _rsi(self.overbought_rsi, "overbought_rsi"),
        )
        if self.oversold_rsi >= self.overbought_rsi:
            raise ValueError("oversold_rsi must be below overbought_rsi")
        object.__setattr__(
            self,
            "stop_atr_multiple",
            _positive(self.stop_atr_multiple, "stop_atr_multiple"),
        )
        object.__setattr__(
            self,
            "minimum_reward_to_risk",
            _positive(self.minimum_reward_to_risk, "minimum_reward_to_risk"),
        )


class MeanReversionStrategy:
    """Fade rejected range extremes and target the rolling statistical mean."""

    def __init__(
        self,
        settings: MeanReversionConfig,
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
        if len(primary.bars) < max(self.settings.band_period, 2):
            return self._hold(context, "mean_reversion_band_warmup_incomplete")

        try:
            indicators = calculate_indicators(primary, self.settings.indicators)
            regime = self.regime_classifier.assess(primary)
        except (IndicatorError, KeyError, ValueError) as exc:
            return self._hold(
                context,
                "mean_reversion_indicator_or_regime_unavailable",
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
                "mean_reversion_market_filter_rejected",
                regime=regime.regime,
                metadata=tuple(
                    (f"filter_{index}", reason)
                    for index, reason in enumerate(filter_result.reasons)
                ),
            )

        closes = primary.closes[-self.settings.band_period :]
        mean = fmean(closes)
        deviation = pstdev(closes)
        if deviation <= 0.0:
            return self._hold(
                context,
                "mean_reversion_zero_band_width",
                regime=regime.regime,
            )
        upper = mean + self.settings.band_standard_deviations * deviation
        lower = mean - self.settings.band_standard_deviations * deviation
        latest = primary.latest
        close = latest.mid.close
        atr = indicators.value("atr")
        rsi = indicators.value("rsi")

        long_candidate = (
            latest.mid.low <= lower
            and close > lower
            and close > latest.mid.open
            and rsi <= self.settings.oversold_rsi
        )
        short_candidate = (
            latest.mid.high >= upper
            and close < upper
            and close < latest.mid.open
            and rsi >= self.settings.overbought_rsi
        )

        action: SignalAction | None = None
        if long_candidate:
            action = SignalAction.BUY
        elif short_candidate:
            action = SignalAction.SELL
        if action is None:
            return self._hold(
                context,
                "mean_reversion_rejection_confirmation_missing",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "band_lower": lower,
                        "band_mean": mean,
                        "band_upper": upper,
                        "close": close,
                        "rsi": rsi,
                    }
                ),
            )

        if action is SignalAction.BUY:
            stop = min(latest.mid.low, lower) - atr * self.settings.stop_atr_multiple
            target = mean
            risk = close - stop
            reward = target - close
        else:
            stop = max(latest.mid.high, upper) + atr * self.settings.stop_atr_multiple
            target = mean
            risk = stop - close
            reward = close - target
        if stop <= 0.0 or target <= 0.0 or risk <= 0.0 or reward <= 0.0:
            return self._hold(
                context,
                "mean_reversion_invalid_price_bracket",
                regime=regime.regime,
            )
        reward_to_risk = reward / risk
        if reward_to_risk < self.settings.minimum_reward_to_risk:
            return self._hold(
                context,
                "mean_reversion_reward_to_risk_below_minimum",
                regime=regime.regime,
                metadata=format_metadata({"reward_to_risk": reward_to_risk}),
            )

        penetration = (
            max(0.0, lower - latest.mid.low)
            if action is SignalAction.BUY
            else max(0.0, latest.mid.high - upper)
        )
        band_width = max(upper - lower, 1e-12)
        penetration_score = min(1.0, penetration / band_width * 4.0)
        rsi_score = (
            min(1.0, (self.settings.oversold_rsi - rsi + 10.0) / 20.0)
            if action is SignalAction.BUY
            else min(1.0, (rsi - self.settings.overbought_rsi + 10.0) / 20.0)
        )
        confidence = clamp_confidence(
            0.35 * max(0.0, penetration_score)
            + 0.30 * max(0.0, rsi_score)
            + 0.20 * min(1.0, reward_to_risk / 2.0)
            + 0.15 * regime.confidence
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
                "range_regime_confirmed",
                "volatility_band_touched",
                "extreme_rejection_candle_confirmed",
                "rsi_exhaustion_confirmed",
            ),
            regime=regime.regime,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            metadata=format_metadata(
                {
                    "atr": atr,
                    "band_lower": lower,
                    "band_mean": mean,
                    "band_upper": upper,
                    "reward_to_risk": reward_to_risk,
                    "rsi": rsi,
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
