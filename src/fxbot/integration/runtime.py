"""End-to-end paper integration runtime for strategy, risk, execution, and health."""

from __future__ import annotations

from datetime import UTC, datetime

from fxbot.execution.broker import PermanentBrokerError
from fxbot.execution.paper import PaperBroker
from fxbot.execution.router import ExecutionRouter
from fxbot.execution.runtime import ExecutionRuntime, SyncResult
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.models import (
    PaperCycleResult,
    PaperFrame,
    PaperOrderOutcome,
)
from fxbot.integration.planner import DecisionOrderPlanner
from fxbot.integration.state import PaperRuntimeStateStore
from fxbot.production.health import ComponentState, HealthRegistry
from fxbot.strategy.base import Strategy, StrategyRuntime


class PaperIntegrationRuntime:
    """Process chronological frames through the complete paper trading stack."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        strategy_runtime: StrategyRuntime,
        planner: DecisionOrderPlanner,
        broker: PaperBroker,
        router: ExecutionRouter,
        execution_runtime: ExecutionRuntime,
        ledger: PaperPortfolioLedger,
        health: HealthRegistry,
        state_store: PaperRuntimeStateStore | None = None,
    ) -> None:
        self.strategy = strategy
        self.strategy_runtime = strategy_runtime
        self.planner = planner
        self.broker = broker
        self.router = router
        self.execution_runtime = execution_runtime
        self.ledger = ledger
        self.health = health
        self.state_store = state_store
        self._cycle = 0
        self._last_frame_at: datetime | None = None
        if state_store is not None:
            state = state_store.load()
            if state is not None:
                self._cycle = state.cycle
                self._last_frame_at = state.last_frame_at
                self.ledger.restore(state)

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def last_frame_at(self) -> datetime | None:
        return self._last_frame_at

    def process(self, frame: PaperFrame) -> PaperCycleResult:
        if self._last_frame_at is not None and frame.quote.timestamp <= self._last_frame_at:
            raise ValueError("paper frames must be strictly chronological")

        self.broker.update_quote(frame.quote)
        self.ledger.on_quote(frame.quote)
        combined_sync = self.execution_runtime.sync()
        decision = self.strategy_runtime.run(self.strategy, frame.context)
        intents = self.planner.plan(decision, frame.quote, self.ledger)
        outcomes: list[PaperOrderOutcome] = []

        for intent in intents:
            try:
                order = self.router.submit(intent)
            except PermanentBrokerError as exc:
                outcomes.append(PaperOrderOutcome(intent=intent, error=str(exc)))
                self.health.update(
                    "risk",
                    ComponentState.DEGRADED,
                    str(exc),
                    checked_at=frame.quote.timestamp,
                )
                continue

            self.execution_runtime.observe_order(order)
            outcomes.append(PaperOrderOutcome(intent=intent, order=order))
            combined_sync = self._combine(combined_sync, self.execution_runtime.sync())

        self.ledger.on_quote(frame.quote)
        self._cycle += 1
        self._last_frame_at = frame.quote.timestamp
        self._update_health(frame, decision.action.value, combined_sync, outcomes)
        if self.state_store is not None:
            self.state_store.save(
                self.ledger.state(
                    cycle=self._cycle,
                    last_frame_at=self._last_frame_at,
                )
            )
        return PaperCycleResult(
            cycle=self._cycle,
            processed_at=frame.quote.timestamp,
            decision=decision,
            outcomes=tuple(outcomes),
            sync=combined_sync,
            account=self.ledger.view(),
        )

    def stop(self) -> None:
        self.health.stop_all("paper runtime stopped")

    @staticmethod
    def _combine(first: SyncResult, second: SyncResult) -> SyncResult:
        return SyncResult(
            new_fills=first.new_fills + second.new_fills,
            duplicate_fills=first.duplicate_fills + second.duplicate_fills,
            order_updates=first.order_updates + second.order_updates,
            warnings=first.warnings + second.warnings,
        )

    def _update_health(
        self,
        frame: PaperFrame,
        action: str,
        sync: SyncResult,
        outcomes: list[PaperOrderOutcome],
    ) -> None:
        now = datetime.now(UTC)
        self.health.update(
            "market_data",
            ComponentState.HEALTHY,
            "paper frame accepted",
            checked_at=now,
            metadata=(("symbol", frame.quote.symbol),),
        )
        self.health.update(
            "strategy",
            ComponentState.HEALTHY,
            f"strategy decision: {action}",
            checked_at=now,
        )
        rejected = sum(not outcome.accepted for outcome in outcomes)
        execution_state = (
            ComponentState.DEGRADED
            if rejected or sync.warnings
            else ComponentState.HEALTHY
        )
        self.health.update(
            "execution",
            execution_state,
            "paper execution cycle completed",
            checked_at=now,
            metadata=(
                ("fills", str(sync.new_fills)),
                ("rejected", str(rejected)),
                ("warnings", str(sync.warnings)),
            ),
        )
        account = self.ledger.view()
        self.health.update(
            "portfolio",
            ComponentState.HEALTHY,
            "paper portfolio marked",
            checked_at=now,
            metadata=(
                ("balance", f"{account.balance:.2f}"),
                ("equity", f"{account.equity:.2f}"),
                ("positions", str(len(account.positions))),
            ),
        )
