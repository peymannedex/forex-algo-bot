import pytest

from fxbot.execution.paper import PaperBroker
from fxbot.execution.router import ExecutionRouter
from fxbot.execution.runtime import ExecutionRuntime
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.planner import DecisionOrderPlanner, FixedQuantityPolicy
from fxbot.integration.runtime import PaperIntegrationRuntime
from fxbot.integration.state import PaperRuntimeStateStore
from fxbot.production.health import HealthRegistry
from fxbot.strategy.base import StrategyRuntime
from fxbot.strategy.models import SignalAction

from .conftest import StaticStrategy, make_frame


def make_runtime(tmp_path, action: SignalAction) -> PaperIntegrationRuntime:
    broker = PaperBroker()
    ledger = PaperPortfolioLedger()
    execution_runtime = ExecutionRuntime(broker, fill_sinks=(ledger,))
    return PaperIntegrationRuntime(
        strategy=StaticStrategy(action),
        strategy_runtime=StrategyRuntime(),
        planner=DecisionOrderPlanner(FixedQuantityPolicy(0.1)),
        broker=broker,
        router=ExecutionRouter(broker),
        execution_runtime=execution_runtime,
        ledger=ledger,
        health=HealthRegistry(),
        state_store=PaperRuntimeStateStore(tmp_path / "state.json"),
    )


def test_buy_cycle_routes_and_consumes_fill(tmp_path) -> None:
    runtime = make_runtime(tmp_path, SignalAction.BUY)

    result = runtime.process(make_frame())

    assert result.accepted_orders == 1
    assert result.sync.new_fills == 1
    assert result.account.positions[0].signed_quantity == pytest.approx(0.1)
    assert runtime.health.snapshot().ready


def test_hold_cycle_does_not_submit(tmp_path) -> None:
    runtime = make_runtime(tmp_path, SignalAction.HOLD)

    result = runtime.process(make_frame())

    assert result.outcomes == ()
    assert result.sync.new_fills == 0


def test_frames_must_be_strictly_chronological(tmp_path) -> None:
    runtime = make_runtime(tmp_path, SignalAction.HOLD)
    frame = make_frame()
    runtime.process(frame)

    with pytest.raises(ValueError, match="chronological"):
        runtime.process(frame)


def test_runtime_restores_cycle_and_position(tmp_path) -> None:
    first = make_runtime(tmp_path, SignalAction.BUY)
    first.process(make_frame())

    second = make_runtime(tmp_path, SignalAction.HOLD)

    assert second.cycle == 1
    assert second.ledger.signed_position("EURUSD") == pytest.approx(0.1)


def test_risk_rejection_is_non_fatal(tmp_path) -> None:
    from fxbot.execution.models import RiskDecision

    class RejectAll:
        def authorize(self, intent):
            return RiskDecision(False, "blocked")

    broker = PaperBroker()
    ledger = PaperPortfolioLedger()
    runtime = PaperIntegrationRuntime(
        strategy=StaticStrategy(SignalAction.BUY),
        strategy_runtime=StrategyRuntime(),
        planner=DecisionOrderPlanner(FixedQuantityPolicy(0.1)),
        broker=broker,
        router=ExecutionRouter(broker, risk_authorizer=RejectAll()),
        execution_runtime=ExecutionRuntime(broker, fill_sinks=(ledger,)),
        ledger=ledger,
        health=HealthRegistry(),
    )

    result = runtime.process(make_frame())

    assert result.rejected_orders == 1
    assert result.account.positions == ()
    assert runtime.health.snapshot().state.value == "degraded"


def test_stop_marks_health_components_stopped(tmp_path) -> None:
    runtime = make_runtime(tmp_path, SignalAction.HOLD)
    runtime.process(make_frame())

    runtime.stop()

    assert runtime.health.snapshot().state.value == "stopped"
