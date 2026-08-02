from __future__ import annotations

from typing import Any

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.strategy.liquidity import LiquidityPool, LiquiditySide, LiquiditySweep
from fxbot.strategy.market_structure import (
    DealingRangeZone,
    DisplacementCandle,
    MarketStructureState,
    StructureConfluence,
    StructureDirection,
    StructureEvent,
    StructureEventKind,
    SwingKind,
    SwingPoint,
)
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyConfig
from fxbot.strategy.smc import SMCStrategy, SMCStrategyConfig


def _strategy_config(*, higher: bool = False) -> StrategyConfig:
    return StrategyConfig(
        strategy_id="smc-test",
        primary_timeframe=Timeframe.M5,
        required_timeframes=(Timeframe.H1,) if higher else (),
        warmup_bars=1,
    )


def _state(context: Any, direction: StructureDirection, *, include_event: bool = True) -> MarketStructureState:
    primary = context.primary
    latest_index = len(primary.bars) - 1
    swing_kind = SwingKind.LOW if direction is StructureDirection.BULLISH else SwingKind.HIGH
    swing_price = primary.latest.mid.close - 0.01 if swing_kind is SwingKind.LOW else primary.latest.mid.close + 0.01
    swing = SwingPoint(
        context.symbol,
        context.primary_timeframe,
        1,
        swing_kind,
        swing_price,
        primary.bars[1].open_time,
        primary.bars[2].close_time,
    )
    events: tuple[StructureEvent, ...] = ()
    if include_event:
        events = (
            StructureEvent(
                context.symbol,
                context.primary_timeframe,
                latest_index,
                StructureEventKind.CHANGE_OF_CHARACTER,
                direction,
                swing.price,
                primary.latest.mid.close,
                primary.latest.close_time,
                swing,
            ),
        )
    displacement = DisplacementCandle(
        context.symbol,
        context.primary_timeframe,
        latest_index,
        direction,
        primary.latest.close_time,
        0.003,
        0.004,
        1.5,
        0.9,
    )
    return MarketStructureState(
        context.symbol,
        context.primary_timeframe,
        primary.latest.close_time,
        direction,
        (swing,),
        events,
        (displacement,),
        None,
        None,
        None,
        DealingRangeZone.UNKNOWN,
    )


def _liquidity(
    context: Any,
    direction: StructureDirection,
    *,
    extreme: float | None = None,
) -> tuple[tuple[LiquidityPool, ...], tuple[LiquiditySweep, ...]]:
    primary = context.primary
    latest_index = len(primary.bars) - 1
    entry = primary.latest.mid.close
    if direction is StructureDirection.BULLISH:
        swept_side = LiquiditySide.SELL_SIDE
        pool_level = entry - 0.004
        sweep_extreme = entry - 0.008 if extreme is None else extreme
        target = LiquidityPool(
            context.symbol,
            context.primary_timeframe,
            LiquiditySide.BUY_SIDE,
            entry + 0.03,
            (3, 5),
            primary.bars[-3].close_time,
            0.0002,
        )
    else:
        swept_side = LiquiditySide.BUY_SIDE
        pool_level = entry + 0.004
        sweep_extreme = entry + 0.008 if extreme is None else extreme
        target = LiquidityPool(
            context.symbol,
            context.primary_timeframe,
            LiquiditySide.SELL_SIDE,
            entry - 0.03,
            (3, 5),
            primary.bars[-3].close_time,
            0.0002,
        )
    swept = LiquidityPool(
        context.symbol,
        context.primary_timeframe,
        swept_side,
        pool_level,
        (2, 4),
        primary.bars[-3].close_time,
        0.0002,
    )
    sweep = LiquiditySweep(
        context.symbol,
        context.primary_timeframe,
        latest_index,
        swept_side,
        direction,
        pool_level,
        sweep_extreme,
        entry,
        abs(pool_level - sweep_extreme),
        primary.latest.close_time,
        swept,
    )
    return (swept, target), (sweep,)


def _patch_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot: Any,
    state: MarketStructureState,
    pools: tuple[LiquidityPool, ...],
    sweeps: tuple[LiquiditySweep, ...],
) -> None:
    monkeypatch.setattr("fxbot.strategy.smc.calculate_indicators", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr("fxbot.strategy.smc.analyze_market_structure", lambda *_args, **_kwargs: state)
    monkeypatch.setattr("fxbot.strategy.smc.detect_liquidity_pools", lambda *_args, **_kwargs: pools)
    monkeypatch.setattr("fxbot.strategy.smc.detect_liquidity_sweeps", lambda *_args, **_kwargs: sweeps)
    monkeypatch.setattr("fxbot.strategy.smc.detect_order_blocks", lambda *_args, **_kwargs: ())
    monkeypatch.setattr("fxbot.strategy.smc.detect_fair_value_gaps", lambda *_args, **_kwargs: ())


def test_smc_config_validation() -> None:
    with pytest.raises(ValueError):
        SMCStrategyConfig(strategy=_strategy_config(), signal_age_bars=-1)
    with pytest.raises(ValueError):
        SMCStrategyConfig(strategy=_strategy_config(), target_reward_to_risk=0)
    with pytest.raises(ValueError):
        SMCStrategyConfig(strategy=_strategy_config(), minimum_structure_alignment=1.1)


def test_smc_generates_bullish_signal(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 + index * 0.0001 for index in range(30)])
    snapshot = make_snapshot(context, atr=0.002)
    state = _state(context, StructureDirection.BULLISH)
    pools, sweeps = _liquidity(context, StructureDirection.BULLISH)
    _patch_components(monkeypatch, snapshot=snapshot, state=state, pools=pools, sweeps=sweeps)
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config()),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.BUY
    assert decision.stop_loss is not None and decision.stop_loss < decision.entry_price
    assert decision.take_profit == pytest.approx(pools[1].level)
    assert "liquidity_pool_swept_and_reclaimed" in decision.reasons


def test_smc_generates_bearish_signal(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 - index * 0.0001 for index in range(30)])
    snapshot = make_snapshot(context, atr=0.002)
    state = _state(context, StructureDirection.BEARISH)
    pools, sweeps = _liquidity(context, StructureDirection.BEARISH)
    _patch_components(monkeypatch, snapshot=snapshot, state=state, pools=pools, sweeps=sweeps)
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config()),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.SELL
    assert decision.stop_loss is not None and decision.stop_loss > decision.entry_price
    assert decision.take_profit == pytest.approx(pools[1].level)


def test_smc_holds_without_recent_sweep(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 + index * 0.0001 for index in range(30)])
    state = _state(context, StructureDirection.BULLISH)
    snapshot = make_snapshot(context)
    _patch_components(monkeypatch, snapshot=snapshot, state=state, pools=(), sweeps=())
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config()),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("smc_recent_liquidity_sweep_missing",)


def test_smc_rejects_missing_structure_shift(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 + index * 0.0001 for index in range(30)])
    state = _state(context, StructureDirection.BULLISH, include_event=False)
    pools, sweeps = _liquidity(context, StructureDirection.BULLISH)
    _patch_components(
        monkeypatch,
        snapshot=make_snapshot(context),
        state=state,
        pools=pools,
        sweeps=sweeps,
    )
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config()),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("smc_confirmation_rejected",)
    assert dict(decision.metadata)["failure_0"] == "structure_shift_missing"


def test_smc_can_require_zone_confluence(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 + index * 0.0001 for index in range(30)])
    state = _state(context, StructureDirection.BULLISH)
    pools, sweeps = _liquidity(context, StructureDirection.BULLISH)
    _patch_components(
        monkeypatch,
        snapshot=make_snapshot(context),
        state=state,
        pools=pools,
        sweeps=sweeps,
    )
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config(), require_zone_confluence=True),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.HOLD
    assert "order_block_or_fvg_confluence_missing" in dict(decision.metadata).values()


def test_smc_market_filter_rejects_illiquid_regime(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 + index * 0.0001 for index in range(30)])
    state = _state(context, StructureDirection.BULLISH)
    pools, sweeps = _liquidity(context, StructureDirection.BULLISH)
    _patch_components(
        monkeypatch,
        snapshot=make_snapshot(context),
        state=state,
        pools=pools,
        sweeps=sweeps,
    )
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config()),
        regime_classifier=classifier_factory(MarketRegime.ILLIQUID),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("smc_market_filter_rejected",)


def test_smc_rejects_opposite_higher_timeframe_structure(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = [1.10 + index * 0.0001 for index in range(30)]
    context = make_context(closes, higher_closes=closes)
    state = _state(context, StructureDirection.BULLISH)
    pools, sweeps = _liquidity(context, StructureDirection.BULLISH)
    _patch_components(
        monkeypatch,
        snapshot=make_snapshot(context),
        state=state,
        pools=pools,
        sweeps=sweeps,
    )
    monkeypatch.setattr(
        "fxbot.strategy.smc.structure_confluence",
        lambda *_args, **_kwargs: StructureConfluence(
            context.symbol,
            context.as_of,
            context.primary_timeframe,
            StructureDirection.BULLISH,
            StructureDirection.BEARISH,
            1.0,
            (
                (Timeframe.M5, StructureDirection.BULLISH),
                (Timeframe.H1, StructureDirection.BEARISH),
            ),
        ),
    )
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config(higher=True)),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.HOLD
    assert "higher_timeframe_structure_mismatch" in dict(decision.metadata).values()


def test_smc_rejects_invalid_stop_geometry(
    make_context: Any,
    make_snapshot: Any,
    classifier_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context([1.10 + index * 0.0001 for index in range(30)])
    state = _state(context, StructureDirection.BULLISH)
    entry = context.primary.latest.mid.close
    pools, sweeps = _liquidity(
        context,
        StructureDirection.BULLISH,
        extreme=entry + 0.01,
    )
    _patch_components(
        monkeypatch,
        snapshot=make_snapshot(context),
        state=state,
        pools=pools,
        sweeps=sweeps,
    )
    strategy = SMCStrategy(
        SMCStrategyConfig(strategy=_strategy_config()),
        regime_classifier=classifier_factory(MarketRegime.UNKNOWN),
    )

    decision = strategy.evaluate(context)

    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ("smc_invalid_stop_geometry",)
