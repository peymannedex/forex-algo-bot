from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.models import (
    IndicatorSnapshot,
    MarketRegime,
    RegimeAssessment,
    RegimeConfluence,
    SignalAction,
    StrategyConfig,
    StrategyDecision,
)

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def test_strategy_config_normalizes_required_timeframes() -> None:
    config = StrategyConfig(
        strategy_id=" trend ",
        primary_timeframe=Timeframe.M5,
        required_timeframes=(Timeframe.H1, Timeframe.M5, Timeframe.H1),
    )
    assert config.strategy_id == "trend"
    assert config.required_timeframes == (Timeframe.M5, Timeframe.H1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy_id": ""}, "strategy_id"),
        ({"primary_timeframe": Timeframe.TICK}, "bar timeframe"),
        ({"warmup_bars": 0}, "warmup_bars"),
        ({"max_data_age": timedelta(0)}, "max_data_age"),
        ({"min_confidence": 1.1}, "min_confidence"),
        ({"duplicate_suppression_window": timedelta(seconds=-1)}, "negative"),
    ],
)
def test_strategy_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    defaults: dict[str, object] = {
        "strategy_id": "s",
        "primary_timeframe": Timeframe.M5,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        StrategyConfig(**defaults)  # type: ignore[arg-type]


def test_decision_normalizes_and_fingerprints_semantics() -> None:
    first = StrategyDecision(
        strategy_id="s",
        symbol="eurusd",
        timeframe=Timeframe.M5,
        action=SignalAction.BUY,
        as_of=NOW,
        confidence=0.8,
        reasons=("trend",),
        regime=MarketRegime.TRENDING_UP,
        entry_price=1.10,
        stop_loss=1.09,
        take_profit=1.12,
        metadata=(("z", "2"), ("a", "1")),
    )
    second = StrategyDecision(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        action=SignalAction.BUY,
        as_of=NOW + timedelta(minutes=5),
        confidence=0.8,
        reasons=("trend",),
        regime=MarketRegime.TRENDING_UP,
        entry_price=1.10,
        stop_loss=1.09,
        take_profit=1.12,
        metadata=(("a", "1"), ("z", "2")),
    )
    assert first.symbol == "EURUSD"
    assert first.metadata == (("a", "1"), ("z", "2"))
    assert first.semantic_fingerprint == second.semantic_fingerprint


def test_decision_fingerprint_changes_with_meaning() -> None:
    buy = StrategyDecision(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        action=SignalAction.BUY,
        as_of=NOW,
        confidence=0.8,
        reasons=("trend",),
    )
    sell = StrategyDecision(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        action=SignalAction.SELL,
        as_of=NOW,
        confidence=0.8,
        reasons=("trend",),
    )
    assert buy.semantic_fingerprint != sell.semantic_fingerprint


def test_hold_factory_builds_non_executable_decision() -> None:
    decision = StrategyDecision.hold(
        strategy_id="s",
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        as_of=NOW,
        reason="warming_up",
    )
    assert decision.action is SignalAction.HOLD
    assert decision.confidence == 0.0
    assert decision.reasons == ("warming_up",)


@pytest.mark.parametrize(
    "decision",
    [
        lambda: StrategyDecision(
            strategy_id="s",
            symbol="EURUSD",
            timeframe=Timeframe.M5,
            action=SignalAction.HOLD,
            as_of=NOW,
            confidence=0.0,
            reasons=("hold",),
            entry_price=1.0,
        ),
        lambda: StrategyDecision(
            strategy_id="s",
            symbol="EURUSD",
            timeframe=Timeframe.M5,
            action=SignalAction.EXIT,
            as_of=NOW,
            confidence=0.5,
            reasons=("exit",),
            stop_loss=1.0,
        ),
        lambda: StrategyDecision(
            strategy_id="s",
            symbol="EURUSD",
            timeframe=Timeframe.M5,
            action=SignalAction.BUY,
            as_of=NOW,
            confidence=0.5,
            reasons=("",),
        ),
    ],
)
def test_decision_rejects_invalid_payloads(decision: object) -> None:
    with pytest.raises(ValueError):
        decision()  # type: ignore[operator]


def test_indicator_snapshot_sorts_and_reads_values() -> None:
    snapshot = IndicatorSnapshot(
        symbol="eurusd",
        timeframe=Timeframe.M5,
        as_of=NOW,
        values=(("rsi", 55.0), ("atr", 0.001)),
        sample_size=100,
    )
    assert snapshot.values == (("atr", 0.001), ("rsi", 55.0))
    assert snapshot.value("rsi") == 55.0
    with pytest.raises(KeyError, match="not available"):
        snapshot.value("missing")


def test_regime_assessment_reads_metric() -> None:
    assessment = RegimeAssessment(
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        as_of=NOW,
        regime=MarketRegime.RANGING,
        confidence=0.7,
        metrics=(("momentum", 0.0),),
        reasons=("compressed",),
    )
    assert assessment.metric("momentum") == 0.0


def test_regime_confluence_directional_alignment() -> None:
    confluence = RegimeConfluence(
        symbol="EURUSD",
        as_of=NOW,
        primary_timeframe=Timeframe.M5,
        primary_regime=MarketRegime.TRENDING_UP,
        dominant_regime=MarketRegime.TRENDING_UP,
        alignment_score=2 / 3,
        assessments=(
            (Timeframe.M5, MarketRegime.TRENDING_UP),
            (Timeframe.H1, MarketRegime.TRENDING_UP),
            (Timeframe.H4, MarketRegime.RANGING),
        ),
    )
    assert confluence.directionally_aligned
