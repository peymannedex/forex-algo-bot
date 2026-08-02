"""Strategy contracts, context, indicators, and regime classification."""

from fxbot.strategy.base import (
    SignalDeduplicator,
    Strategy,
    StrategyContractError,
    StrategyRuntime,
)
from fxbot.strategy.context import (
    ContextIssue,
    ContextIssueCode,
    MarketContextBuilder,
    MarketSeries,
    MultiTimeframeContext,
    timeframe_age_tolerance,
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
from fxbot.strategy.models import (
    IndicatorSnapshot,
    MarketRegime,
    RegimeAssessment,
    RegimeConfluence,
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)
from fxbot.strategy.regime import RegimeClassifier, RegimeConfig

__all__ = [
    "ContextIssue",
    "ContextIssueCode",
    "IndicatorConfig",
    "IndicatorError",
    "IndicatorSnapshot",
    "MarketContextBuilder",
    "MarketRegime",
    "MarketSeries",
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
    "StrategyRuntime",
    "average_spread",
    "average_true_range",
    "calculate_indicators",
    "exponential_moving_average",
    "momentum",
    "realized_volatility",
    "relative_strength_index",
    "simple_moving_average",
    "timeframe_age_tolerance",
    "true_range",
]
