"""Momentum-continuation strategy with RSI, EMA, and impulse confirmation."""

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
        allowed_regimes=(
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.VOLATILE,
        ),
        max_spread_to_atr=0.25,
        minimum_alignment_score=0.50,
    )


@dataclass(frozen=True, slots=True)
class MomentumStrategyConfig:
    """Thresholds for directional momentum continuation signals."""

    strategy: StrategyConfig
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    filters: StrategyFilterConfig = field(default_factory=_default_filters)
    minimum_momentum: float = 0.0010
    minimum_impulse_atr: float = 0.20
    long_rsi_minimum: float = 55.0
    long_rsi_maximum: float = 78.0
    short_rsi_minimum: float = 22.0
    short_rsi_maximum: float = 45.0
    stop_atr_multiple: float = 1.25
    target_atr_multiple: float = 2.50
    require_higher_timeframe_confirmation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_momentum",
            _positive(self.minimum_momentum, "minimum_momentum"),
        )
        object.__setattr__(
            self,
            "minimum_impulse_atr",
            _non_negative(self.minimum_impulse_atr, "minimum_impulse_atr"),
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
        if self.require_higher_timeframe_confirmation and len(
            self.strategy.required_timeframes
        ) < 2:
            raise ValueError(
                "Higher-timeframe confirmation requires another timeframe"
            )


class MomentumStrategy:
    """Trade directional acceleration when momentum and trend agree."""

    def __init__(
        self,
        settings: MomentumStrategyConfig,
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
            return self._hold(context, "momentum_requires_two_primary_bars")

        try:
            indicators = calculate_indicators(primary, self.settings.indicators)
            regime = self.regime_classifier.assess(primary)
            confluence = self._confluence(context)
        except (IndicatorError, KeyError, ValueError) as exc:
            return self._hold(
                context,
                "momentum_indicator_or_regime_unavailable",
                metadata=(("error", str(exc)),),
            )

        latest = primary.latest
        previous = primary.bars[-2]
        close = latest.mid.close
        previous_close = previous.mid.close
        atr = indicators.value("atr")
        momentum_value = indicators.value("momentum")
        rsi = indicators.value("rsi")
        fast = indicators.value("fast_ema")
        slow = indicators.value("slow_ema")
        impulse_atr = abs(close - previous_close) / max(atr, 1e-12)

        long_candidate = (
            momentum_value >= self.settings.minimum_momentum
            and impulse_atr >= self.settings.minimum_impulse_atr
            and fast >= slow
            and close > previous_close
            and latest.mid.close > latest.mid.open
            and self.settings.long_rsi_minimum <= rsi <= self.settings.long_rsi_maximum
            and regime.regime is not MarketRegime.TRENDING_DOWN
        )
        short_candidate = (
            momentum_value <= -self.settings.minimum_momentum
            and impulse_atr >= self.settings.minimum_impulse_atr
            and fast <= slow
            and close < previous_close
            and latest.mid.close < latest.mid.open
            and self.settings.short_rsi_minimum <= rsi <= self.settings.short_rsi_maximum
            and regime.regime is not MarketRegime.TRENDING_UP
        )

        action: SignalAction | None = None
        if long_candidate:
            action = SignalAction.BUY
        elif short_candidate:
            action = SignalAction.SELL

        if action is not None and not self._direction_confirmed(action, confluence):
            return self._hold(
                context,
                "momentum_higher_timeframe_not_confirmed",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "alignment": (
                            confluence.alignment_score if confluence is not None else 0.0
                        ),
                        "momentum": momentum_value,
                    }
                ),
            )

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
                "momentum_market_filter_rejected",
                regime=regime.regime,
                metadata=tuple(
                    (f"filter_{index}", reason)
                    for index, reason in enumerate(filter_result.reasons)
                ),
            )
        if action is None:
            return self._hold(
                context,
                "momentum_thresholds_not_met",
                regime=regime.regime,
                metadata=format_metadata(
                    {
                        "fast_ema": fast,
                        "impulse_atr": impulse_atr,
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
        momentum_score = min(
            1.0,
            abs(momentum_value) / (self.settings.minimum_momentum * 3.0),
        )
        impulse_score = min(
            1.0,
            impulse_atr / max(self.settings.minimum_impulse_atr * 3.0, 1e-12),
        )
        rsi_score = _rsi_direction_score(action, rsi)
        alignment_score = confluence.alignment_score if confluence is not None else 1.0
        confidence = clamp_confidence(
            0.35 * momentum_score
            + 0.25 * impulse_score
            + 0.20 * rsi_score
            + 0.10 * alignment_score
            + 0.10 * regime.confidence
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
                "momentum_threshold_confirmed",
                "rsi_direction_confirmed",
                "ema_direction_confirmed",
                "price_impulse_confirmed",
            ),
            regime=regime.regime,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            metadata=format_metadata(
                {
                    "atr": atr,
                    "impulse_atr": impulse_atr,
                    "momentum": momentum_value,
                    "regime_alignment": alignment_score,
                    "rsi": rsi,
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

    def _direction_confirmed(
        self,
        action: SignalAction,
        confluence: RegimeConfluence | None,
    ) -> bool:
        if not self.settings.require_higher_timeframe_confirmation:
            return True
        if confluence is None:
            return False
        if action is SignalAction.BUY:
            return confluence.dominant_regime is MarketRegime.TRENDING_UP
        return confluence.dominant_regime is MarketRegime.TRENDING_DOWN

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


def _rsi_direction_score(action: SignalAction, rsi: float) -> float:
    center = 65.0 if action is SignalAction.BUY else 35.0
    return max(0.0, 1.0 - abs(rsi - center) / 35.0)
