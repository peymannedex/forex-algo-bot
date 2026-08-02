"""Strategy contracts, context, indicators, regimes, filters, and implementations."""

from fxbot.strategy.base import (
    SignalDeduplicator,
    Strategy,
    StrategyContractError,
    StrategyRuntime,
)
from fxbot.strategy.breakout import BreakoutStrategy, BreakoutStrategyConfig
from fxbot.strategy.context import (
    ContextIssue,
    ContextIssueCode,
    MarketContextBuilder,
    MarketSeries,
    MultiTimeframeContext,
    timeframe_age_tolerance,
)
from fxbot.strategy.filters import (
    FilterResult,
    MarketFilter,
    StrategyFilterConfig,
    atr_bracket,
    clamp_confidence,
    format_metadata,
)
from fxbot.strategy.indicators import (
    IndicatorConfig,
    IndicatorError,
    RollingIndicatorState,
    average_spread,
    average_true_range,
    calculate_indicators,
    exponential_moving_average,
    momentum,
    realized_volatility,
    relative_strength_index,
    simple_moving_average,
    true_range,
)
from fxbot.strategy.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from fxbot.strategy.models import (
    IndicatorSnapshot,
    MarketRegime,
    RegimeAssessment,
    RegimeConfluence,
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)
from fxbot.strategy.momentum import MomentumStrategy, MomentumStrategyConfig
from fxbot.strategy.regime import RegimeClassifier, RegimeConfig
from fxbot.strategy.trend_following import TrendFollowingConfig, TrendFollowingStrategy

__all__ = [
    "BreakoutStrategy",
    "BreakoutStrategyConfig",
    "ContextIssue",
    "ContextIssueCode",
    "FilterResult",
    "IndicatorConfig",
    "IndicatorError",
    "IndicatorSnapshot",
    "MarketContextBuilder",
    "MarketFilter",
    "MarketRegime",
    "MarketSeries",
    "MeanReversionConfig",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MomentumStrategyConfig",
    "MultiTimeframeContext",
    "RegimeAssessment",
    "RegimeClassifier",
    "RegimeConfig",
    "RegimeConfluence",
    "RollingIndicatorState",
    "SignalAction",
    "SignalDeduplicator",
    "Strategy",
    "StrategyConfig",
    "StrategyContractError",
    "StrategyDecision",
    "StrategyFilterConfig",
    "StrategyRuntime",
    "TrendFollowingConfig",
    "TrendFollowingStrategy",
    "atr_bracket",
    "average_spread",
    "average_true_range",
    "calculate_indicators",
    "clamp_confidence",
    "exponential_moving_average",
    "format_metadata",
    "momentum",
    "realized_volatility",
    "relative_strength_index",
    "simple_moving_average",
    "timeframe_age_tolerance",
    "true_range",
]
