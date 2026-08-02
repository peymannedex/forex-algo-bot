"""Deterministic Forex position sizing with risk, broker, and margin caps."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal
from math import isfinite
from typing import Protocol

from fxbot.risk.models import (
    InstrumentRiskSpec,
    PositionSizingPolicy,
    PositionSizingRequest,
    PositionSizingResult,
    SizingMethod,
    SizingStatus,
)


class CurrencyConversionError(RuntimeError):
    """Raised when a required account-currency conversion is unavailable."""


class CurrencyConverter(Protocol):
    """Structural conversion contract used by the position-sizing engine."""

    def rate(self, from_currency: str, to_currency: str) -> float: ...


class IdentityCurrencyConverter:
    """Converter for instruments already denominated in account currency."""

    def rate(self, from_currency: str, to_currency: str) -> float:
        source = from_currency.strip().upper()
        target = to_currency.strip().upper()
        if source != target:
            raise CurrencyConversionError(
                f"No conversion rate configured for {source}/{target}"
            )
        return 1.0


class StaticCurrencyConverter:
    """Deterministic direct/inverse conversion table for tests and snapshots."""

    def __init__(self, rates: Mapping[tuple[str, str], float]) -> None:
        normalized: dict[tuple[str, str], float] = {}
        for (source, target), value in rates.items():
            pair = (source.strip().upper(), target.strip().upper())
            number = float(value)
            if not pair[0] or not pair[1]:
                raise ValueError("Currency codes cannot be empty")
            if not isfinite(number) or number <= 0.0:
                raise ValueError(f"Invalid conversion rate for {pair}: {value!r}")
            normalized[pair] = number
        self._rates = normalized

    def rate(self, from_currency: str, to_currency: str) -> float:
        source = from_currency.strip().upper()
        target = to_currency.strip().upper()
        if source == target:
            return 1.0
        direct = self._rates.get((source, target))
        if direct is not None:
            return direct
        inverse = self._rates.get((target, source))
        if inverse is not None:
            return 1.0 / inverse
        raise CurrencyConversionError(
            f"No conversion rate configured for {source}/{target}"
        )


def convert_amount(
    amount: float,
    from_currency: str,
    to_currency: str,
    converter: CurrencyConverter,
) -> float:
    """Convert a non-negative monetary amount and validate the returned rate."""

    value = float(amount)
    if not isfinite(value) or value < 0.0:
        raise ValueError("amount must be finite and non-negative")
    rate = float(converter.rate(from_currency, to_currency))
    if not isfinite(rate) or rate <= 0.0:
        raise CurrencyConversionError(
            f"Converter returned invalid rate for {from_currency}/{to_currency}: {rate!r}"
        )
    return value * rate


def risk_per_lot(
    instrument: InstrumentRiskSpec,
    price_distance: float,
    account_currency: str,
    converter: CurrencyConverter,
) -> float:
    """Return stop-loss value for one lot in account currency."""

    distance = float(price_distance)
    if not isfinite(distance) or distance <= 0.0:
        raise ValueError("price_distance must be a positive finite number")

    if instrument.tick_value is not None:
        assert instrument.tick_size is not None
        assert instrument.tick_value_currency is not None
        amount = distance / instrument.tick_size * instrument.tick_value
        currency = instrument.tick_value_currency
    else:
        amount = distance * instrument.symbol.contract_size
        currency = instrument.symbol.quote_currency

    return convert_amount(amount, currency, account_currency, converter)


def margin_per_lot(
    instrument: InstrumentRiskSpec,
    entry_price: float,
    account_currency: str,
    leverage: float,
    converter: CurrencyConverter,
) -> float:
    """Estimate required margin for one lot in account currency."""

    price = float(entry_price)
    if not isfinite(price) or price <= 0.0:
        raise ValueError("entry_price must be a positive finite number")
    leverage_value = float(leverage)
    if not isfinite(leverage_value) or leverage_value < 1.0:
        raise ValueError("leverage must be finite and at least 1")

    quote_notional = instrument.symbol.contract_size * price
    quote_margin = (
        quote_notional * instrument.margin_rate
        if instrument.margin_rate is not None
        else quote_notional / leverage_value
    )
    return convert_amount(
        quote_margin,
        instrument.symbol.quote_currency,
        account_currency,
        converter,
    )


def normalize_volume_down(
    volume: float,
    instrument: InstrumentRiskSpec,
) -> float:
    """Round volume down to the broker grid without increasing exposure."""

    requested = float(volume)
    if not isfinite(requested) or requested < 0.0:
        raise ValueError("volume must be finite and non-negative")

    constraints = instrument.volume
    if requested < constraints.minimum:
        return 0.0

    maximum = Decimal(str(constraints.maximum))
    minimum = Decimal(str(constraints.minimum))
    step = Decimal(str(constraints.step))
    # Absorb only sub-picovolume floating-point noise before flooring.
    tolerance = step * Decimal("1e-12")
    candidate = min(Decimal(str(requested)) + tolerance, maximum)
    increments = ((candidate - minimum) / step).to_integral_value(rounding=ROUND_FLOOR)
    normalized = minimum + increments * step
    if normalized < minimum:
        return 0.0
    if normalized > maximum:
        normalized = maximum
    return float(normalized)


class PositionSizer:
    """Calculate executable lot size under account and broker constraints."""

    def __init__(
        self,
        *,
        policy: PositionSizingPolicy | None = None,
        converter: CurrencyConverter | None = None,
    ) -> None:
        self.policy = policy or PositionSizingPolicy()
        self.converter = converter or IdentityCurrencyConverter()

    def size(self, request: PositionSizingRequest) -> PositionSizingResult:
        """Return a deterministic, auditable position-sizing decision."""

        if (
            request.risk_fraction is not None
            and request.risk_fraction > self.policy.max_risk_fraction
        ):
            raise ValueError(
                "risk_fraction exceeds the configured max_risk_fraction"
            )

        stop_distance = request.stop_distance
        effective_distance = stop_distance
        if request.method is SizingMethod.VOLATILITY_ADJUSTED:
            assert request.volatility_distance is not None
            effective_distance = max(
                stop_distance,
                request.volatility_distance * request.volatility_multiplier,
            )

        account = request.account
        instrument = request.instrument
        per_lot_risk = risk_per_lot(
            instrument,
            effective_distance,
            account.currency,
            self.converter,
        )
        per_lot_margin = margin_per_lot(
            instrument,
            request.entry_price,
            account.currency,
            account.leverage,
            self.converter,
        )
        if per_lot_risk <= 0.0 or per_lot_margin <= 0.0:
            raise ValueError("Calculated per-lot risk and margin must be positive")

        maximum_risk = account.equity * self.policy.max_risk_fraction
        maximum_margin = min(
            account.free_margin,
            account.equity * self.policy.max_margin_fraction,
        )

        if request.method is SizingMethod.FIXED_VOLUME:
            assert request.fixed_volume is not None
            requested_volume = request.fixed_volume
            requested_risk = requested_volume * per_lot_risk
        else:
            assert request.risk_fraction is not None
            requested_risk = account.equity * request.risk_fraction
            requested_volume = requested_risk / per_lot_risk

        limiting: list[str] = []
        capped_volume = requested_volume

        risk_cap = maximum_risk / per_lot_risk
        if capped_volume > risk_cap:
            capped_volume = risk_cap
            limiting.append("risk_limit")

        margin_cap = maximum_margin / per_lot_margin
        if capped_volume > margin_cap:
            capped_volume = margin_cap
            limiting.append("margin_limit")

        if capped_volume > instrument.volume.maximum:
            capped_volume = instrument.volume.maximum
            limiting.append("broker_maximum")

        normalized = normalize_volume_down(capped_volume, instrument)
        if normalized < capped_volume and normalized > 0.0:
            limiting.append("volume_step")

        stop_pips = stop_distance / instrument.symbol.pip_size
        if normalized <= 0.0:
            return PositionSizingResult(
                status=SizingStatus.REJECTED,
                method=request.method,
                symbol=instrument.symbol.symbol,
                account_currency=account.currency,
                requested_volume=requested_volume,
                capped_volume=max(capped_volume, 0.0),
                normalized_volume=0.0,
                stop_distance=stop_distance,
                effective_stop_distance=effective_distance,
                stop_pips=stop_pips,
                risk_per_lot=per_lot_risk,
                requested_risk_amount=requested_risk,
                maximum_risk_amount=maximum_risk,
                final_risk_amount=0.0,
                margin_per_lot=per_lot_margin,
                maximum_margin_amount=maximum_margin,
                final_margin_amount=0.0,
                limiting_factors=tuple(limiting),
                rejection_reason=(
                    "Available risk, margin, or requested volume is below the broker minimum"
                ),
            )

        final_risk = normalized * per_lot_risk
        final_margin = normalized * per_lot_margin
        return PositionSizingResult(
            status=SizingStatus.ACCEPTED,
            method=request.method,
            symbol=instrument.symbol.symbol,
            account_currency=account.currency,
            requested_volume=requested_volume,
            capped_volume=capped_volume,
            normalized_volume=normalized,
            stop_distance=stop_distance,
            effective_stop_distance=effective_distance,
            stop_pips=stop_pips,
            risk_per_lot=per_lot_risk,
            requested_risk_amount=requested_risk,
            maximum_risk_amount=maximum_risk,
            final_risk_amount=final_risk,
            margin_per_lot=per_lot_margin,
            maximum_margin_amount=maximum_margin,
            final_margin_amount=final_margin,
            limiting_factors=tuple(limiting),
        )
