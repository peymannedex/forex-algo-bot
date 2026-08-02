from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.broker import BrokerSnapshot
from fxbot.backtest.config import BacktestConfig, InstrumentConfig
from fxbot.backtest.engine import (
    BacktestEngine,
    RiskGateResult,
    StrategyDecisionAdapter,
)
from fxbot.backtest.events import MarketEvent, OrderRequest, OrderSide, OrderType, SimulatedFill
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import Tick
from fxbot.strategy.models import MarketRegime, SignalAction, StrategyDecision

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def records() -> tuple[Tick, ...]:
    return tuple(
        Tick(
            symbol="EURUSD",
            event_time=BASE + timedelta(seconds=index),
            bid=1.1000 + index * 0.001,
            ask=1.1002 + index * 0.001,
        )
        for index in range(3)
    )


def config(*, liquidate: bool = False) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=10_000,
        instruments=(InstrumentConfig("EURUSD", contract_size=100_000),),
        liquidate_at_end=liquidate,
    )


class BuyOnce:
    def on_market(
        self,
        event: MarketEvent,
        snapshot: BrokerSnapshot,
    ) -> tuple[OrderRequest, ...]:
        del snapshot
        if event.sequence != 0:
            return ()
        return (
            OrderRequest(
                order_id="entry",
                symbol=event.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                volume=1.0,
                submitted_at=event.timestamp,
            ),
        )


class RejectAll:
    def evaluate(self, order: OrderRequest, snapshot: BrokerSnapshot) -> RiskGateResult:
        del order, snapshot
        return RiskGateResult(False, "portfolio_limit")


class Observer:
    def __init__(self) -> None:
        self.fills: list[SimulatedFill] = []

    def on_fill(self, fill: SimulatedFill, snapshot: BrokerSnapshot) -> None:
        assert snapshot.timestamp == fill.timestamp
        self.fills.append(fill)


def test_engine_defers_strategy_order_until_next_market_event() -> None:
    result = BacktestEngine(config()).run(records(), BuyOnce())
    assert len(result.fills) == 1
    assert result.fills[0].timestamp == BASE + timedelta(seconds=1)
    assert result.fills[0].price == pytest.approx(1.1012)
    assert result.final_snapshot.positions[0].volume == pytest.approx(1.0)


def test_engine_applies_risk_gate_before_broker_submission() -> None:
    result = BacktestEngine(config(), risk_gate=RejectAll()).run(records(), BuyOnce())
    assert result.fills == ()
    assert result.rejected_order_count == 1
    assert result.orders[0].rejection_reason == "portfolio_limit"


def test_engine_notifies_fill_observers() -> None:
    observer = Observer()
    BacktestEngine(config(), fill_observers=(observer,)).run(records(), BuyOnce())
    assert len(observer.fills) == 1
    assert observer.fills[0].order_id == "entry"


def test_engine_liquidates_positions_at_end() -> None:
    result = BacktestEngine(config(liquidate=True)).run(records(), BuyOnce())
    assert result.final_snapshot.positions == ()
    assert len(result.trades) == 1
    assert len(result.fills) == 2
    assert result.final_equity == pytest.approx(result.final_snapshot.balance)


def test_strategy_decision_adapter_maps_buy_and_exit() -> None:
    decisions = {
        0: SignalAction.BUY,
        1: SignalAction.HOLD,
        2: SignalAction.EXIT,
    }

    def provider(event: MarketEvent, snapshot: BrokerSnapshot) -> StrategyDecision:
        del snapshot
        action = decisions[event.sequence]
        if action is SignalAction.HOLD:
            return StrategyDecision.hold(
                strategy_id="adapter",
                symbol=event.symbol,
                timeframe=Timeframe.M1,
                as_of=event.timestamp,
                reason="wait",
            )
        return StrategyDecision(
            strategy_id="adapter",
            symbol=event.symbol,
            timeframe=Timeframe.M1,
            action=action,
            as_of=event.timestamp,
            confidence=0.8,
            reasons=("test",),
            regime=MarketRegime.TRENDING_UP,
            entry_price=1.1 if action is SignalAction.BUY else None,
            stop_loss=1.09 if action is SignalAction.BUY else None,
            take_profit=1.12 if action is SignalAction.BUY else None,
        )

    adapter = StrategyDecisionAdapter(provider, fixed_volume=0.5)
    result = BacktestEngine(config()).run(records(), adapter)
    assert [fill.side for fill in result.fills] == [OrderSide.BUY]
    # EXIT is submitted at the final event and cannot fill without a later event.
    assert any(order.request.reduce_only for order in result.orders)


def test_engine_rejects_order_with_wrong_event_timestamp() -> None:
    class BadStrategy:
        def on_market(
            self,
            event: MarketEvent,
            snapshot: BrokerSnapshot,
        ) -> tuple[OrderRequest, ...]:
            del snapshot
            return (
                OrderRequest(
                    order_id=f"bad-{event.sequence}",
                    symbol=event.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    volume=1,
                    submitted_at=event.timestamp + timedelta(seconds=1),
                ),
            )

    with pytest.raises(ValueError, match="submitted_at"):
        BacktestEngine(config()).run(records(), BadStrategy())
