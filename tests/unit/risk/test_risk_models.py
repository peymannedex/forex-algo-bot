from __future__ import annotations

import pytest

from fxbot.domain.models import SymbolSpec
from fxbot.risk.models import (
    AccountSnapshot,
    BrokerVolumeConstraints,
    InstrumentRiskSpec,
    PositionSizingPolicy,
    PositionSizingRequest,
    SizingMethod,
    TradeSide,
)


def symbol() -> SymbolSpec:
    return SymbolSpec(
        symbol="eurusd",
        base_currency="eur",
        quote_currency="usd",
        digits=5,
        point_size=0.00001,
        pip_size=0.0001,
    )


def account() -> AccountSnapshot:
    return AccountSnapshot(
        currency="usd",
        balance=10_000,
        equity=10_000,
        free_margin=8_000,
        margin_used=2_000,
        leverage=100,
    )


def instrument() -> InstrumentRiskSpec:
    return InstrumentRiskSpec(
        symbol=symbol(),
        volume=BrokerVolumeConstraints(minimum=0.01, maximum=100, step=0.01),
    )


def test_account_snapshot_normalizes_currency_and_values() -> None:
    snapshot = account()
    assert snapshot.currency == "USD"
    assert snapshot.equity == 10_000
    assert snapshot.leverage == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("balance", -1),
        ("equity", 0),
        ("free_margin", -1),
        ("margin_used", -1),
        ("leverage", 0.5),
    ],
)
def test_account_snapshot_rejects_invalid_values(field: str, value: float) -> None:
    values: dict[str, object] = {
        "currency": "USD",
        "balance": 10_000,
        "equity": 10_000,
        "free_margin": 8_000,
        "margin_used": 2_000,
        "leverage": 100,
    }
    values[field] = value
    with pytest.raises(ValueError):
        AccountSnapshot(**values)  # type: ignore[arg-type]


def test_broker_volume_constraints_validate_bounds() -> None:
    constraints = BrokerVolumeConstraints(minimum=0.01, maximum=10, step=0.01)
    assert constraints.minimum == 0.01
    with pytest.raises(ValueError, match="maximum"):
        BrokerVolumeConstraints(minimum=1, maximum=0.5, step=0.1)
    with pytest.raises(ValueError, match="step"):
        BrokerVolumeConstraints(minimum=0.01, maximum=1, step=2)


def test_instrument_defaults_tick_size_to_point_size() -> None:
    spec = instrument()
    assert spec.tick_size == spec.symbol.point_size
    assert spec.tick_value is None
    assert spec.tick_value_currency is None


def test_instrument_tick_value_defaults_to_quote_currency() -> None:
    spec = InstrumentRiskSpec(
        symbol=symbol(),
        volume=BrokerVolumeConstraints(0.01, 10, 0.01),
        tick_size=0.00001,
        tick_value=1,
    )
    assert spec.tick_value_currency == "USD"


def test_instrument_rejects_currency_without_tick_value() -> None:
    with pytest.raises(ValueError, match="requires tick_value"):
        InstrumentRiskSpec(
            symbol=symbol(),
            volume=BrokerVolumeConstraints(0.01, 10, 0.01),
            tick_value_currency="USD",
        )


def test_instrument_rejects_margin_rate_above_one() -> None:
    with pytest.raises(ValueError, match="cannot exceed 1"):
        InstrumentRiskSpec(
            symbol=symbol(),
            volume=BrokerVolumeConstraints(0.01, 10, 0.01),
            margin_rate=1.01,
        )


def test_position_sizing_policy_validates_fractions() -> None:
    policy = PositionSizingPolicy(max_risk_fraction=0.03, max_margin_fraction=0.4)
    assert policy.max_risk_fraction == 0.03
    with pytest.raises(ValueError):
        PositionSizingPolicy(max_risk_fraction=0)
    with pytest.raises(ValueError):
        PositionSizingPolicy(max_margin_fraction=1.1)


def test_long_request_requires_stop_below_entry() -> None:
    with pytest.raises(ValueError, match="long position"):
        PositionSizingRequest(
            account=account(),
            instrument=instrument(),
            side=TradeSide.LONG,
            entry_price=1.10,
            stop_price=1.11,
            risk_fraction=0.01,
        )


def test_short_request_requires_stop_above_entry() -> None:
    with pytest.raises(ValueError, match="short position"):
        PositionSizingRequest(
            account=account(),
            instrument=instrument(),
            side=TradeSide.SHORT,
            entry_price=1.10,
            stop_price=1.09,
            risk_fraction=0.01,
        )


def test_fixed_fractional_request_requires_risk_fraction() -> None:
    with pytest.raises(ValueError, match="risk_fraction is required"):
        PositionSizingRequest(
            account=account(),
            instrument=instrument(),
            side=TradeSide.LONG,
            entry_price=1.10,
            stop_price=1.09,
        )


def test_fixed_volume_request_requires_fixed_volume() -> None:
    with pytest.raises(ValueError, match="fixed_volume is required"):
        PositionSizingRequest(
            account=account(),
            instrument=instrument(),
            side=TradeSide.LONG,
            entry_price=1.10,
            stop_price=1.09,
            method=SizingMethod.FIXED_VOLUME,
        )


def test_volatility_adjusted_request_requires_distance() -> None:
    with pytest.raises(ValueError, match="volatility_distance is required"):
        PositionSizingRequest(
            account=account(),
            instrument=instrument(),
            side=TradeSide.LONG,
            entry_price=1.10,
            stop_price=1.09,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            risk_fraction=0.01,
        )


def test_request_reports_stop_distance() -> None:
    request = PositionSizingRequest(
        account=account(),
        instrument=instrument(),
        side=TradeSide.SHORT,
        entry_price=1.10,
        stop_price=1.105,
        risk_fraction=0.01,
    )
    assert request.stop_distance == pytest.approx(0.005)
