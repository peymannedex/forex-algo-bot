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
    SizingStatus,
    TradeSide,
)
from fxbot.risk.position_sizing import (
    CurrencyConversionError,
    IdentityCurrencyConverter,
    PositionSizer,
    StaticCurrencyConverter,
    convert_amount,
    margin_per_lot,
    normalize_volume_down,
    risk_per_lot,
)


def eurusd() -> InstrumentRiskSpec:
    return InstrumentRiskSpec(
        symbol=SymbolSpec(
            symbol="EURUSD",
            base_currency="EUR",
            quote_currency="USD",
            digits=5,
            point_size=0.00001,
            pip_size=0.0001,
            contract_size=100_000,
        ),
        volume=BrokerVolumeConstraints(minimum=0.01, maximum=100, step=0.01),
    )


def account(
    *,
    currency: str = "USD",
    equity: float = 10_000,
    free_margin: float = 8_000,
    leverage: float = 100,
) -> AccountSnapshot:
    return AccountSnapshot(
        currency=currency,
        balance=equity,
        equity=equity,
        free_margin=free_margin,
        leverage=leverage,
    )


def test_static_converter_supports_direct_inverse_and_identity() -> None:
    converter = StaticCurrencyConverter({("USD", "EUR"): 0.8})
    assert converter.rate("USD", "EUR") == 0.8
    assert converter.rate("EUR", "USD") == 1.25
    assert converter.rate("USD", "USD") == 1.0


def test_static_converter_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        StaticCurrencyConverter({("USD", "EUR"): 0})


def test_missing_conversion_raises_domain_error() -> None:
    converter = IdentityCurrencyConverter()
    with pytest.raises(CurrencyConversionError):
        converter.rate("JPY", "USD")


def test_convert_amount_rejects_invalid_converter_result() -> None:
    class BadConverter:
        def rate(self, from_currency: str, to_currency: str) -> float:
            return 0.0

    with pytest.raises(CurrencyConversionError):
        convert_amount(100, "USD", "EUR", BadConverter())


def test_risk_per_lot_uses_contract_size_for_fx_pair() -> None:
    result = risk_per_lot(eurusd(), 0.005, "USD", IdentityCurrencyConverter())
    assert result == pytest.approx(500)


def test_risk_per_lot_converts_quote_currency() -> None:
    usdjpy = InstrumentRiskSpec(
        symbol=SymbolSpec(
            symbol="USDJPY",
            base_currency="USD",
            quote_currency="JPY",
            digits=3,
            point_size=0.001,
            pip_size=0.01,
            contract_size=100_000,
        ),
        volume=BrokerVolumeConstraints(0.01, 100, 0.01),
    )
    converter = StaticCurrencyConverter({("JPY", "USD"): 0.0066666667})
    result = risk_per_lot(usdjpy, 0.5, "USD", converter)
    assert result == pytest.approx(333.333335)


def test_risk_per_lot_uses_explicit_tick_value() -> None:
    gold = InstrumentRiskSpec(
        symbol=SymbolSpec(
            symbol="XAUUSD",
            base_currency="XAU",
            quote_currency="USD",
            digits=2,
            point_size=0.01,
            pip_size=0.1,
            contract_size=100,
        ),
        volume=BrokerVolumeConstraints(0.01, 50, 0.01),
        tick_size=0.01,
        tick_value=1.0,
        tick_value_currency="USD",
    )
    assert risk_per_lot(gold, 2.0, "USD", IdentityCurrencyConverter()) == 200


def test_margin_per_lot_uses_leverage() -> None:
    result = margin_per_lot(eurusd(), 1.10, "USD", 100, IdentityCurrencyConverter())
    assert result == pytest.approx(1_100)


def test_margin_per_lot_uses_explicit_margin_rate() -> None:
    instrument = InstrumentRiskSpec(
        symbol=eurusd().symbol,
        volume=eurusd().volume,
        margin_rate=0.05,
    )
    result = margin_per_lot(instrument, 1.10, "USD", 500, IdentityCurrencyConverter())
    assert result == pytest.approx(5_500)


def test_normalize_volume_down_uses_broker_grid() -> None:
    instrument = eurusd()
    assert normalize_volume_down(0.237, instrument) == 0.23
    assert normalize_volume_down(0.01, instrument) == 0.01
    assert normalize_volume_down(0.009, instrument) == 0.0


def test_fixed_fractional_sizes_standard_eurusd_trade() -> None:
    request = PositionSizingRequest(
        account=account(),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
        risk_fraction=0.01,
    )
    result = PositionSizer().size(request)
    assert result.accepted
    assert result.status is SizingStatus.ACCEPTED
    assert result.stop_pips == pytest.approx(50)
    assert result.risk_per_lot == pytest.approx(500)
    assert result.normalized_volume == pytest.approx(0.20)
    assert result.final_risk_amount == pytest.approx(100)


def test_fixed_fractional_supports_short_trade() -> None:
    request = PositionSizingRequest(
        account=account(),
        instrument=eurusd(),
        side=TradeSide.SHORT,
        entry_price=1.1000,
        stop_price=1.1050,
        risk_fraction=0.01,
    )
    result = PositionSizer().size(request)
    assert result.normalized_volume == pytest.approx(0.20)


def test_volume_is_rounded_down_never_up() -> None:
    request = PositionSizingRequest(
        account=account(equity=12_345),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
        risk_fraction=0.01,
    )
    result = PositionSizer().size(request)
    assert result.requested_volume == pytest.approx(0.2469)
    assert result.normalized_volume == 0.24
    assert "volume_step" in result.limiting_factors
    assert result.final_risk_amount <= result.requested_risk_amount


def test_fixed_volume_is_reduced_by_risk_cap() -> None:
    request = PositionSizingRequest(
        account=account(),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0900,
        method=SizingMethod.FIXED_VOLUME,
        fixed_volume=1.0,
    )
    result = PositionSizer(policy=PositionSizingPolicy(max_risk_fraction=0.02)).size(request)
    assert result.normalized_volume == pytest.approx(0.20)
    assert "risk_limit" in result.limiting_factors
    assert result.final_risk_amount == pytest.approx(200)


def test_margin_cap_reduces_volume() -> None:
    request = PositionSizingRequest(
        account=account(free_margin=100, leverage=100),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0990,
        risk_fraction=0.01,
    )
    result = PositionSizer().size(request)
    assert "margin_limit" in result.limiting_factors
    assert result.normalized_volume == pytest.approx(0.09)
    assert result.final_margin_amount == pytest.approx(99)


def test_broker_maximum_caps_volume() -> None:
    capped_instrument = InstrumentRiskSpec(
        symbol=eurusd().symbol,
        volume=BrokerVolumeConstraints(0.01, 0.10, 0.01),
    )
    request = PositionSizingRequest(
        account=account(equity=100_000, free_margin=100_000),
        instrument=capped_instrument,
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0990,
        risk_fraction=0.01,
    )
    result = PositionSizer().size(request)
    assert result.normalized_volume == 0.10
    assert "broker_maximum" in result.limiting_factors


def test_below_minimum_volume_is_rejected() -> None:
    request = PositionSizingRequest(
        account=account(equity=100, free_margin=100),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0900,
        risk_fraction=0.001,
    )
    result = PositionSizer(
        policy=PositionSizingPolicy(max_risk_fraction=0.01)
    ).size(request)
    assert not result.accepted
    assert result.status is SizingStatus.REJECTED
    assert result.normalized_volume == 0
    assert result.rejection_reason is not None


def test_volatility_adjustment_uses_larger_effective_stop() -> None:
    request = PositionSizingRequest(
        account=account(),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0980,
        method=SizingMethod.VOLATILITY_ADJUSTED,
        risk_fraction=0.01,
        volatility_distance=0.003,
        volatility_multiplier=2.0,
    )
    result = PositionSizer().size(request)
    assert result.stop_distance == pytest.approx(0.002)
    assert result.effective_stop_distance == pytest.approx(0.006)
    assert result.normalized_volume == pytest.approx(0.16)
    assert result.final_risk_amount <= 100


def test_account_currency_conversion_changes_risk_and_margin() -> None:
    converter = StaticCurrencyConverter({("USD", "EUR"): 0.90})
    request = PositionSizingRequest(
        account=account(currency="EUR"),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
        risk_fraction=0.01,
    )
    result = PositionSizer(converter=converter).size(request)
    assert result.risk_per_lot == pytest.approx(450)
    assert result.margin_per_lot == pytest.approx(990)
    assert result.normalized_volume == pytest.approx(0.22)


def test_risk_fraction_above_policy_is_rejected() -> None:
    request = PositionSizingRequest(
        account=account(),
        instrument=eurusd(),
        side=TradeSide.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
        risk_fraction=0.03,
    )
    with pytest.raises(ValueError, match="exceeds"):
        PositionSizer(policy=PositionSizingPolicy(max_risk_fraction=0.02)).size(request)
