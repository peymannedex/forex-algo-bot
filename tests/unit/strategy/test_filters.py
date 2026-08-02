from __future__ import annotations

from datetime import time

import pytest

from fxbot.strategy.filters import (
    MarketFilter,
    StrategyFilterConfig,
    atr_bracket,
    clamp_confidence,
    format_metadata,
)
from fxbot.strategy.models import MarketRegime, SignalAction


def test_filter_allows_clean_market(make_context, make_snapshot, classifier_factory) -> None:
    context = make_context([1.10] * 30)
    snapshot = make_snapshot(context)
    regime = classifier_factory(MarketRegime.TRENDING_UP).assess(context.primary)
    result = MarketFilter().evaluate(context=context, indicators=snapshot, regime=regime)
    assert result.allowed
    assert result.confidence_multiplier > 0.0


def test_filter_rejects_spread(make_context, make_snapshot, classifier_factory) -> None:
    context = make_context([1.10] * 30)
    snapshot = make_snapshot(context, spread_to_atr=0.5)
    regime = classifier_factory(MarketRegime.TRENDING_UP).assess(context.primary)
    result = MarketFilter(StrategyFilterConfig(max_spread_to_atr=0.2)).evaluate(
        context=context,
        indicators=snapshot,
        regime=regime,
    )
    assert not result.allowed
    assert "spread_to_atr_exceeds_limit" in result.reasons


def test_filter_rejects_blocked_regime(make_context, make_snapshot, classifier_factory) -> None:
    context = make_context([1.10] * 30)
    snapshot = make_snapshot(context)
    regime = classifier_factory(MarketRegime.ILLIQUID).assess(context.primary)
    result = MarketFilter().evaluate(context=context, indicators=snapshot, regime=regime)
    assert not result.allowed
    assert "blocked_regime:illiquid" in result.reasons


def test_filter_requires_directional_confluence(
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context([1.10] * 30, higher_closes=[1.10] * 30)
    snapshot = make_snapshot(context)
    classifier = classifier_factory(
        MarketRegime.TRENDING_UP,
        dominant=MarketRegime.TRENDING_DOWN,
    )
    regime = classifier.assess(context.primary)
    confluence = classifier.confluence(context)
    config = StrategyFilterConfig(
        require_directional_alignment=True,
        minimum_alignment_score=0.5,
    )
    result = MarketFilter(config).evaluate(
        context=context,
        indicators=snapshot,
        regime=regime,
        confluence=confluence,
        expected_action=SignalAction.BUY,
    )
    assert not result.allowed
    assert "regime_direction_mismatch" in result.reasons


def test_overnight_session_is_supported(
    make_context,
    make_snapshot,
    classifier_factory,
) -> None:
    context = make_context([1.10] * 30)
    snapshot = make_snapshot(context)
    regime = classifier_factory(MarketRegime.TRENDING_UP).assess(context.primary)
    config = StrategyFilterConfig(session_start=time(22), session_end=time(8))
    result = MarketFilter(config).evaluate(
        context=context,
        indicators=snapshot,
        regime=regime,
    )
    assert not result.allowed
    assert "outside_session" in result.reasons


def test_atr_bracket_is_directional() -> None:
    long_stop, long_target = atr_bracket(
        action=SignalAction.BUY,
        entry_price=1.10,
        atr=0.01,
        stop_multiple=1.0,
        target_multiple=2.0,
    )
    short_stop, short_target = atr_bracket(
        action=SignalAction.SELL,
        entry_price=1.10,
        atr=0.01,
        stop_multiple=1.0,
        target_multiple=2.0,
    )
    assert long_stop < 1.10 < long_target
    assert short_target < 1.10 < short_stop


def test_helpers_validate_and_sort() -> None:
    assert clamp_confidence(1.5) == 1.0
    assert clamp_confidence(-1.0) == 0.0
    assert format_metadata({"z": 2.0, "a": 1}) == (("a", "1"), ("z", "2"))
    with pytest.raises(ValueError):
        StrategyFilterConfig(session_start=time(8))
    with pytest.raises(ValueError):
        atr_bracket(
            action=SignalAction.HOLD,
            entry_price=1.0,
            atr=0.1,
            stop_multiple=1.0,
            target_multiple=2.0,
        )
