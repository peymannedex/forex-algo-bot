import pytest

from fxbot.integration.factory import default_instrument
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.planner import PositionSizerQuantityPolicy
from fxbot.risk.position_sizing import IdentityCurrencyConverter, PositionSizer
from fxbot.strategy.models import SignalAction

from .conftest import StaticStrategy, make_frame


def test_position_sizer_policy_returns_broker_normalized_volume() -> None:
    frame = make_frame()
    decision = StaticStrategy(
        SignalAction.BUY,
        stop_loss=1.0990,
        take_profit=1.1030,
    ).evaluate(frame.context)
    ledger = PaperPortfolioLedger()
    ledger.on_quote(frame.quote)
    policy = PositionSizerQuantityPolicy(
        sizer=PositionSizer(converter=IdentityCurrencyConverter()),
        instruments={"EURUSD": default_instrument("EURUSD")},
        risk_fraction=0.005,
    )

    quantity = policy.quantity(decision, frame.quote, ledger)

    assert quantity > 0.0
    assert quantity * 100 == pytest.approx(round(quantity * 100))


def test_position_sizer_policy_requires_stop() -> None:
    frame = make_frame()
    decision = StaticStrategy(SignalAction.BUY).evaluate(frame.context)
    policy = PositionSizerQuantityPolicy(
        sizer=PositionSizer(converter=IdentityCurrencyConverter()),
        instruments={"EURUSD": default_instrument("EURUSD")},
    )

    assert policy.quantity(decision, frame.quote, PaperPortfolioLedger()) == 0.0


def test_position_sizer_policy_requires_instrument() -> None:
    frame = make_frame()
    decision = StaticStrategy(
        SignalAction.BUY,
        stop_loss=1.0990,
    ).evaluate(frame.context)
    policy = PositionSizerQuantityPolicy(
        sizer=PositionSizer(converter=IdentityCurrencyConverter()),
        instruments={},
    )

    with pytest.raises(KeyError, match="No risk instrument"):
        policy.quantity(decision, frame.quote, PaperPortfolioLedger())
