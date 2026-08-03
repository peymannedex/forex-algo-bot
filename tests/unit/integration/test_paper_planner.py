from fxbot.execution.models import ExecutionFill, OrderSide
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.planner import DecisionOrderPlanner, FixedQuantityPolicy
from fxbot.strategy.models import SignalAction

from .conftest import StaticStrategy, make_frame


def evaluate(action: SignalAction):
    frame = make_frame()
    return StaticStrategy(action).evaluate(frame.context), frame


def test_hold_plans_no_orders() -> None:
    decision, frame = evaluate(SignalAction.HOLD)
    planner = DecisionOrderPlanner(FixedQuantityPolicy(0.1))

    assert planner.plan(decision, frame.quote, PaperPortfolioLedger()) == ()


def test_buy_plans_market_entry() -> None:
    decision, frame = evaluate(SignalAction.BUY)
    planner = DecisionOrderPlanner(FixedQuantityPolicy(0.1))

    intents = planner.plan(decision, frame.quote, PaperPortfolioLedger())

    assert len(intents) == 1
    assert intents[0].side is OrderSide.BUY
    assert not intents[0].reduce_only
    assert intents[0].quantity == 0.1


def test_exit_plans_reduce_only_order() -> None:
    decision, frame = evaluate(SignalAction.EXIT)
    ledger = PaperPortfolioLedger()
    ledger.on_quote(frame.quote)
    ledger.on_fill(
        ExecutionFill(
            "fill-1",
            "order-1",
            "client-1",
            "EURUSD",
            OrderSide.BUY,
            0.2,
            frame.quote.ask,
            frame.quote.timestamp,
        )
    )
    planner = DecisionOrderPlanner(FixedQuantityPolicy(0.1))

    intents = planner.plan(decision, frame.quote, ledger)

    assert len(intents) == 1
    assert intents[0].side is OrderSide.SELL
    assert intents[0].reduce_only
    assert intents[0].quantity == 0.2


def test_opposite_signal_plans_exit_then_entry() -> None:
    decision, frame = evaluate(SignalAction.SELL)
    ledger = PaperPortfolioLedger()
    ledger.on_quote(frame.quote)
    ledger.on_fill(
        ExecutionFill(
            "fill-1",
            "order-1",
            "client-1",
            "EURUSD",
            OrderSide.BUY,
            0.2,
            frame.quote.ask,
            frame.quote.timestamp,
        )
    )
    planner = DecisionOrderPlanner(FixedQuantityPolicy(0.1))

    intents = planner.plan(decision, frame.quote, ledger)

    assert [intent.reduce_only for intent in intents] == [True, False]
    assert [intent.side for intent in intents] == [OrderSide.SELL, OrderSide.SELL]


def test_same_direction_is_not_pyramided() -> None:
    decision, frame = evaluate(SignalAction.BUY)
    ledger = PaperPortfolioLedger()
    ledger.on_quote(frame.quote)
    ledger.on_fill(
        ExecutionFill(
            "fill-1",
            "order-1",
            "client-1",
            "EURUSD",
            OrderSide.BUY,
            0.1,
            frame.quote.ask,
            frame.quote.timestamp,
        )
    )
    planner = DecisionOrderPlanner(FixedQuantityPolicy(0.1))

    assert planner.plan(decision, frame.quote, ledger) == ()
