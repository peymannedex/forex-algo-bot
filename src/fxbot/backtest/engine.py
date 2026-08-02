"""Event-driven backtest orchestration and strategy/risk integration seams."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from fxbot.backtest.broker import BrokerSnapshot, SimulatedBroker
from fxbot.backtest.clock import HistoricalClock
from fxbot.backtest.config import BacktestConfig
from fxbot.backtest.events import (
    AuditEvent,
    EventKind,
    MarketDataRecord,
    MarketEvent,
    OrderRequest,
    OrderSide,
    OrderType,
    SimulatedFill,
)
from fxbot.backtest.results import BacktestResult, EquityPoint, equity_point
from fxbot.strategy.models import SignalAction, StrategyDecision


class BacktestStrategy(Protocol):
    """Strategy adapter invoked after each market event has been processed."""

    def on_market(
        self,
        event: MarketEvent,
        snapshot: BrokerSnapshot,
    ) -> Iterable[OrderRequest]: ...


class FillObserver(Protocol):
    """Optional integration hook for Phase 2 fill reconciliation."""

    def on_fill(self, fill: SimulatedFill, snapshot: BrokerSnapshot) -> None: ...


@dataclass(frozen=True, slots=True)
class RiskGateResult:
    """Portable approval result used by the engine before broker submission."""

    approved: bool
    reason: str = "approved"
    metadata: tuple[tuple[str, str], ...] = ()


class RiskGate(Protocol):
    """Portable interface implemented by Phase 2 risk-guard adapters."""

    def evaluate(self, order: OrderRequest, snapshot: BrokerSnapshot) -> RiskGateResult: ...


class AllowAllRiskGate:
    """Default risk gate for isolated broker and strategy tests."""

    def evaluate(self, order: OrderRequest, snapshot: BrokerSnapshot) -> RiskGateResult:
        del order, snapshot
        return RiskGateResult(approved=True)


DecisionProvider = Callable[[MarketEvent, BrokerSnapshot], StrategyDecision]
VolumeProvider = Callable[[StrategyDecision, BrokerSnapshot], float]


class StrategyDecisionAdapter:
    """Translate canonical Phase 3 decisions into simulated market orders.

    Entries are submitted after the decision event and therefore fill no earlier
    than the next market event. EXIT decisions flatten the current symbol only.
    Stop and target proposals remain in order metadata for later lifecycle or
    bracket-order orchestration.
    """

    def __init__(
        self,
        decision_provider: DecisionProvider,
        *,
        fixed_volume: float | None = None,
        volume_provider: VolumeProvider | None = None,
    ) -> None:
        if fixed_volume is None and volume_provider is None:
            raise ValueError("fixed_volume or volume_provider is required")
        if fixed_volume is not None and volume_provider is not None:
            raise ValueError("Specify only one of fixed_volume or volume_provider")
        if fixed_volume is not None and fixed_volume <= 0.0:
            raise ValueError("fixed_volume must be positive")
        self.decision_provider = decision_provider
        self.fixed_volume = fixed_volume
        self.volume_provider = volume_provider

    def on_market(
        self,
        event: MarketEvent,
        snapshot: BrokerSnapshot,
    ) -> tuple[OrderRequest, ...]:
        decision = self.decision_provider(event, snapshot)
        if decision.symbol != event.symbol:
            raise ValueError("StrategyDecision symbol does not match market event")
        if decision.as_of != event.timestamp:
            raise ValueError("StrategyDecision as_of must match market event timestamp")
        if decision.action is SignalAction.HOLD:
            return ()

        position = snapshot.position(event.symbol)
        if decision.action is SignalAction.EXIT:
            if position is None:
                return ()
            side = OrderSide.SELL if position.signed_volume > 0.0 else OrderSide.BUY
            volume = position.volume
            reduce_only = True
        else:
            side = OrderSide.BUY if decision.action is SignalAction.BUY else OrderSide.SELL
            volume = self._volume(decision, snapshot)
            reduce_only = False

        metadata = [
            ("confidence", f"{decision.confidence:.12g}"),
            ("regime", decision.regime.value),
            ("reasons", " | ".join(decision.reasons)),
        ]
        if decision.stop_loss is not None:
            metadata.append(("proposed_stop_loss", f"{decision.stop_loss:.12g}"))
        if decision.take_profit is not None:
            metadata.append(("proposed_take_profit", f"{decision.take_profit:.12g}"))
        metadata.extend(decision.metadata)

        return (
            OrderRequest(
                order_id=(
                    f"{decision.strategy_id}:{event.sequence}:"
                    f"{decision.action.value}:{event.symbol}"
                ),
                symbol=event.symbol,
                side=side,
                order_type=OrderType.MARKET,
                volume=volume,
                submitted_at=event.timestamp,
                reduce_only=reduce_only,
                client_tag=decision.strategy_id,
                metadata=tuple(metadata),
            ),
        )

    def _volume(self, decision: StrategyDecision, snapshot: BrokerSnapshot) -> float:
        if self.volume_provider is not None:
            volume = float(self.volume_provider(decision, snapshot))
        else:
            assert self.fixed_volume is not None
            volume = self.fixed_volume
        if volume <= 0.0:
            raise ValueError("Strategy volume must be positive")
        return volume


class BacktestEngine:
    """Run a deterministic market-data, strategy, risk, and broker event loop."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        risk_gate: RiskGate | None = None,
        fill_observers: tuple[FillObserver, ...] = (),
    ) -> None:
        self.config = config
        self.risk_gate = risk_gate or AllowAllRiskGate()
        self.fill_observers = fill_observers

    def run(
        self,
        records: Iterable[MarketDataRecord],
        strategy: BacktestStrategy,
    ) -> BacktestResult:
        """Replay historical records without same-event strategy fills."""

        clock = HistoricalClock(
            records,
            strict_input_order=self.config.strict_input_order,
        )
        events = clock.replay()
        if not events:
            raise ValueError("Backtest requires at least one market record")

        broker = SimulatedBroker(self.config)
        engine_audit: list[AuditEvent] = []
        equity_curve: list[EquityPoint] = []
        running_peak = self.config.initial_cash

        for event in events:
            engine_audit.append(
                AuditEvent(
                    sequence=event.sequence,
                    timestamp=event.timestamp,
                    kind=EventKind.MARKET,
                    message="Historical market event replayed",
                    metadata=(("symbol", event.symbol),),
                )
            )
            fills = broker.on_market(event)
            self._notify_fills(fills, broker)

            snapshot = broker.snapshot(event.timestamp)
            for order in tuple(strategy.on_market(event, snapshot)):
                self._validate_order_event(order, event)
                engine_audit.append(
                    AuditEvent(
                        sequence=event.sequence,
                        timestamp=event.timestamp,
                        kind=EventKind.ORDER_SUBMITTED,
                        message="Strategy submitted order",
                        order_id=order.order_id,
                    )
                )
                decision = self.risk_gate.evaluate(order, snapshot)
                if decision.approved:
                    broker.submit(order, current_sequence=event.sequence)
                else:
                    broker.reject(
                        order,
                        current_sequence=event.sequence,
                        reason=decision.reason,
                    )

            point = equity_point(broker.snapshot(event.timestamp), running_peak)
            running_peak = max(running_peak, point.equity)
            equity_curve.append(point)

        final_event = events[-1]
        if self.config.liquidate_at_end and broker.snapshot(final_event.timestamp).positions:
            liquidation_fills = broker.liquidate(
                timestamp=final_event.timestamp,
                sequence=final_event.sequence + 1,
            )
            self._notify_fills(liquidation_fills, broker)
            final_point = equity_point(broker.snapshot(final_event.timestamp), running_peak)
            equity_curve.append(final_point)

        final_snapshot = broker.snapshot(final_event.timestamp)
        audit = tuple(
            sorted(
                (*engine_audit, *broker.audit_events),
                key=lambda item: (
                    item.timestamp,
                    item.sequence,
                    item.kind.value,
                    item.order_id or "",
                    item.fill_id or "",
                ),
            )
        )
        return BacktestResult(
            config=self.config,
            started_at=events[0].timestamp,
            ended_at=final_event.timestamp,
            final_snapshot=final_snapshot,
            orders=broker.orders,
            fills=broker.fills,
            trades=broker.trades,
            equity_curve=tuple(equity_curve),
            audit_events=audit,
        )

    def _notify_fills(
        self,
        fills: tuple[SimulatedFill, ...],
        broker: SimulatedBroker,
    ) -> None:
        for fill in fills:
            snapshot = broker.snapshot(fill.timestamp)
            for observer in self.fill_observers:
                observer.on_fill(fill, snapshot)

    @staticmethod
    def _validate_order_event(order: OrderRequest, event: MarketEvent) -> None:
        if order.submitted_at != event.timestamp:
            raise ValueError("Order submitted_at must match the current market event")
        if order.symbol != event.symbol:
            raise ValueError("Order symbol must match the current market event")
