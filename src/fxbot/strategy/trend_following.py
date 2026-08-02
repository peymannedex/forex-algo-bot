"""EMA-aligned trend-following strategy with pullback confirmation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from fxbot.strategy.context import MultiTimeframeContext
from fxbot.strategy.filters import (
    MarketFilter,
    StrategyFilterConfig,
    atr_bracket,
    clamp_confidence,
    format_metadata,
)
from fxbot.strategy.indicators import IndicatorConfig, IndicatorError, calculate_indicators
from fxbot.strategy.models import (
    MarketRegime,
    RegimeConfluence,
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)
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
        allowed_regimes=(MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN),
        max_spread_to_atr=0.25,
        minimum_alignment_score=0.50,
        require_directional_alignment=True,
    )


@dataclass(frozen=True, slots=True)
class TrendFollowingConfig:
    """Parameters for trend alignment, pullback entry, and ATR risk brackets."""

    strategy: StrategyConfig
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    filters: StrategyFilterConfig = field(default_factory=_default_filters)
    pullback_tolerance_atr: float = 0.40
    minimum_momentum: float = 0.0005
    long_rsi_minimum: float = 48.0
    long_rsi_maximum: float = 72.0
    short_rsi_minimum: float = 28.0
    short_rsi_maximum: float = 52.0
    stop_atr_multiple: float = 1.50
    target_atr_multiple: float = 3.00

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pullback_tolerance_atr",
            _non_negative(self.pullback_tolerance_atr, "pullback_tolerance_atr"),
        )
        object.__setattr__(
            self,
            "minimum_momentum",
            _non_negative(self.minimum_momentum, "minimum_momentum"),
        )
        for name in (
            "long_rsi_minimum",
            "long_rsi_maximum",
            "short_rsi_minimum",
            "short_rsi_maximum",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100")
        if self.long_rsi_minimum > self.long_rsi_maximum:
            raise ValueError("long_rsi_minimum cannot exceed long_rsi_maximum")
        if self.short_rsi_minimum > self.short_rsi_maximum:
            raise ValueError("short_rsi_minimum cannot exceed short_rsi_maximum")
        object.__setattr__(
            self,
            "stop_atr_multiple",
            _positive(self.stop_atr_multiple, "stop_atr_multiple"),
        )
        object.__setattr__(
            self,
            "target_atr_multiple",
            _positive(self.target_atr_multiple, "target_atr_multiple"),
        )
        if self.filters.require_directional_alignment and len(
            self.strategy.required_timeframes
        ) < 2:
            raise ValueError(
                "Directional alignment requires at least one higher timeframe"
            )


class TrendFollowingStrategy:
    """Enter trend continuation after an EMA pullback and candle confirmation."""

    def __init__(
        self,
        settings: TrendFollowingConfig,
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
        if len(primary.bars) < 2:
            return self._hold(context, "trend_requires_two_primary_bars")

        try:
            indicators = calculate_indicators(primary, self.settings.indicators)
            regime = self.regime_classifier.assess(primary)
            confluence = self._confluence(context)
        except (IndicatorError, KeyError, ValueError) as exc:
            return self._hold(
                context,
                "trend_indicator_or_regime_unavailable",
                metadata=(("error", str(exc)),),
            )

        latest = primary.latest
        previous = primary.bars[-2]
        close = latest.mid.close
        fast = indicators.value("fast_ema")
        slow = indicators.value("slow_ema")
        atr = indicators.value("atr")
        momentum_value = indicators.value("momentum")
        rsi = indicators.value("rsi")
        tolerance = atr * self.settings.pullback_tolerance_atr

        long_candidate = (
            regime.regime is MarketRegime.TRENDING_UP
            and fast > slow
            and close >= slow
            and latest.mid.low <= fast + tolerance
            and close >= fast - tolerance
            and latest.mid.close > latest.mid.open
            and close >= previous.mid.close
            and momentum_value >= self.settings.minimum_momentum
            and self.settings.long_rsi_minimum <= rsi <= self.settings.long_rsi_maximum
        )
        short_candidate = (
            regime.regime is MarketRegime.TRENDING_DOWN
            and fast < slow
            and close <= slow
            and latest.mid.high >= fast - tolerance
            and close <= fast + tolerance
            and latest.mid.close < latest.mid.open
            and close <= previous.mid.close
            and momentum_value <= -self.settings.minimum_momentum
            and self.settings.short_rsi_minimum <= rsi <= self.settings.short_rsi_maximum
        )

        action: SignalAction | None = None
        if long_candidate:
            action = SignalAction.BUY
        elif short_candidate:
            action = SignalAction.SELL

        filter_result = self.market_filter.evaluate(
            context=context,
            indicators=indicators,
            regime=regime,
            confluence=confluence,
            expected_action=action,
        )
        if not filter_result.allowed:
            return self._hold(
                context,
                "trend_market_filter_rejected",
                regime=regime.regime,
                metadata=tuple(
                    (f"filter_{index}", reason)
                    for index, reason in enumerate(filter_result.reasons)
                ),
            )
        if action is None:
            return self._hold(
                context,
                "trend_pullback_confirmation_missing",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "atr": atr,
                        "fast_ema": fast,
                        "momentum": momentum_value,
                        "rsi": rsi,
                        "slow_ema": slow,
                    }
                ),
            )

        stop, target = atr_bracket(
            action=action,
            entry_price=close,
            atr=atr,
            stop_multiple=self.settings.stop_atr_multiple,
            target_multiple=self.settings.target_atr_multiple,
        )
        ema_strength = min(1.0, abs(fast - slow) / max(atr, 1e-12))
        momentum_strength = min(
            1.0,
            abs(momentum_value) / max(self.settings.minimum_momentum * 3.0, 1e-12),
        )
        pullback_quality = 1.0 - min(
            1.0,
            abs(close - fast) / max(tolerance, atr * 0.05, 1e-12),
        )
        confidence = clamp_confidence(
            (0.35 * ema_strength + 0.30 * momentum_strength + 0.20 * pullback_quality)
            + 0.15 * regime.confidence
        )
        confidence *= filter_result.confidence_multiplier

        reasons = (
            "ema_trend_alignment",
            "pullback_reached_fast_ema_zone",
            "trend_candle_confirmation",
            "momentum_and_rsi_confirmed",
        )
        return StrategyDecision(
            strategy_id=self.config.strategy_id,
            symbol=context.symbol,
            timeframe=context.primary_timeframe,
            action=action,
            as_of=context.as_of,
            confidence=clamp_confidence(confidence),
            reasons=reasons,
            regime=regime.regime,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            metadata=format_metadata(
                {
                    "atr": atr,
                    "fast_ema": fast,
                    "momentum": momentum_value,
                    "regime_alignment": (
                        confluence.alignment_score if confluence is not None else 1.0
                    ),
                    "rsi": rsi,
                    "slow_ema": slow,
                }
            ),
        )

    def _confluence(self, context: MultiTimeframeContext) -> RegimeConfluence | None:
        if len(self.config.required_timeframes) < 2:
            return None
        return self.regime_classifier.confluence(
            context,
            self.config.required_timeframes,
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
