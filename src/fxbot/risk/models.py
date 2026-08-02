"""Immutable risk-domain models used by sizing and portfolio controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from fxbot.domain.models import SymbolSpec


def _currency(value: str, field_name: str) -> str:
    normalized = value.strip().upper()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError(f"{field_name} must be a non-empty alphanumeric currency code")
    if len(normalized) > 12:
        raise ValueError(f"{field_name} is too long")
    return normalized


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return number


class TradeSide(StrEnum):
    """Directional exposure of a proposed position."""

    LONG = "long"
    SHORT = "short"


class SizingMethod(StrEnum):
    """Supported position-sizing algorithms."""

    FIXED_FRACTIONAL = "fixed_fractional"
    FIXED_VOLUME = "fixed_volume"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


class SizingStatus(StrEnum):
    """Whether a sizing request produced an executable broker volume."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Point-in-time account values required by the risk engine."""

    currency: str
    balance: float
    equity: float
    free_margin: float
    margin_used: float = 0.0
    leverage: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _currency(self.currency, "currency"))
        object.__setattr__(self, "balance", _non_negative(self.balance, "balance"))
        object.__setattr__(self, "equity", _positive(self.equity, "equity"))
        object.__setattr__(
            self,
            "free_margin",
            _non_negative(self.free_margin, "free_margin"),
        )
        object.__setattr__(
            self,
            "margin_used",
            _non_negative(self.margin_used, "margin_used"),
        )
        object.__setattr__(self, "leverage", _positive(self.leverage, "leverage"))
        if self.leverage < 1.0:
            raise ValueError("leverage must be at least 1")


@dataclass(frozen=True, slots=True)
class BrokerVolumeConstraints:
    """Broker lot-size bounds and increment for one trading instrument."""

    minimum: float
    maximum: float
    step: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum", _positive(self.minimum, "minimum"))
        object.__setattr__(self, "maximum", _positive(self.maximum, "maximum"))
        object.__setattr__(self, "step", _positive(self.step, "step"))
        if self.maximum < self.minimum:
            raise ValueError("maximum volume cannot be below minimum volume")
        if self.step > self.maximum:
            raise ValueError("volume step cannot exceed maximum volume")


@dataclass(frozen=True, slots=True)
class InstrumentRiskSpec:
    """Broker and contract metadata needed for monetary risk calculations.

    ``tick_value`` is the monetary value of one ``tick_size`` move for one lot,
    denominated in ``tick_value_currency``.  When omitted, the engine derives
    value from the FX contract size and quote currency.
    """

    symbol: SymbolSpec
    volume: BrokerVolumeConstraints
    tick_size: float | None = None
    tick_value: float | None = None
    tick_value_currency: str | None = None
    margin_rate: float | None = None

    def __post_init__(self) -> None:
        tick_size = self.symbol.point_size if self.tick_size is None else _positive(
            self.tick_size,
            "tick_size",
        )
        object.__setattr__(self, "tick_size", tick_size)

        if self.tick_value is None:
            if self.tick_value_currency is not None:
                raise ValueError("tick_value_currency requires tick_value")
        else:
            object.__setattr__(self, "tick_value", _positive(self.tick_value, "tick_value"))
            currency = self.tick_value_currency or self.symbol.quote_currency
            object.__setattr__(
                self,
                "tick_value_currency",
                _currency(currency, "tick_value_currency"),
            )

        if self.margin_rate is not None:
            rate = _positive(self.margin_rate, "margin_rate")
            if rate > 1.0:
                raise ValueError("margin_rate cannot exceed 1")
            object.__setattr__(self, "margin_rate", rate)


@dataclass(frozen=True, slots=True)
class PositionSizingPolicy:
    """Hard account-level limits applied to every sizing request."""

    max_risk_fraction: float = 0.02
    max_margin_fraction: float = 0.50

    def __post_init__(self) -> None:
        risk = _positive(self.max_risk_fraction, "max_risk_fraction")
        margin = _positive(self.max_margin_fraction, "max_margin_fraction")
        if risk > 1.0:
            raise ValueError("max_risk_fraction cannot exceed 1")
        if margin > 1.0:
            raise ValueError("max_margin_fraction cannot exceed 1")
        object.__setattr__(self, "max_risk_fraction", risk)
        object.__setattr__(self, "max_margin_fraction", margin)


@dataclass(frozen=True, slots=True)
class PositionSizingRequest:
    """One proposed position with entry, protective stop, and sizing method."""

    account: AccountSnapshot
    instrument: InstrumentRiskSpec
    side: TradeSide
    entry_price: float
    stop_price: float
    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_fraction: float | None = None
    fixed_volume: float | None = None
    volatility_distance: float | None = None
    volatility_multiplier: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", TradeSide(self.side))
        object.__setattr__(self, "method", SizingMethod(self.method))
        entry = _positive(self.entry_price, "entry_price")
        stop = _positive(self.stop_price, "stop_price")
        object.__setattr__(self, "entry_price", entry)
        object.__setattr__(self, "stop_price", stop)

        if self.side is TradeSide.LONG and stop >= entry:
            raise ValueError("A long position requires stop_price below entry_price")
        if self.side is TradeSide.SHORT and stop <= entry:
            raise ValueError("A short position requires stop_price above entry_price")

        if self.method is SizingMethod.FIXED_VOLUME:
            if self.fixed_volume is None:
                raise ValueError("fixed_volume is required for fixed-volume sizing")
            object.__setattr__(
                self,
                "fixed_volume",
                _positive(self.fixed_volume, "fixed_volume"),
            )
            if self.risk_fraction is not None:
                raise ValueError("risk_fraction is not used for fixed-volume sizing")
        else:
            if self.risk_fraction is None:
                raise ValueError("risk_fraction is required for risk-based sizing")
            fraction = _positive(self.risk_fraction, "risk_fraction")
            if fraction > 1.0:
                raise ValueError("risk_fraction cannot exceed 1")
            object.__setattr__(self, "risk_fraction", fraction)
            if self.fixed_volume is not None:
                raise ValueError("fixed_volume is only valid for fixed-volume sizing")

        if self.method is SizingMethod.VOLATILITY_ADJUSTED:
            if self.volatility_distance is None:
                raise ValueError(
                    "volatility_distance is required for volatility-adjusted sizing"
                )
            object.__setattr__(
                self,
                "volatility_distance",
                _positive(self.volatility_distance, "volatility_distance"),
            )
            object.__setattr__(
                self,
                "volatility_multiplier",
                _positive(self.volatility_multiplier, "volatility_multiplier"),
            )
        elif self.volatility_distance is not None:
            raise ValueError(
                "volatility_distance is only valid for volatility-adjusted sizing"
            )

    @property
    def stop_distance(self) -> float:
        """Absolute executable price distance between entry and stop."""

        return abs(self.entry_price - self.stop_price)


@dataclass(frozen=True, slots=True)
class PositionSizingResult:
    """Auditable output of the deterministic position-sizing engine."""

    status: SizingStatus
    method: SizingMethod
    symbol: str
    account_currency: str
    requested_volume: float
    capped_volume: float
    normalized_volume: float
    stop_distance: float
    effective_stop_distance: float
    stop_pips: float
    risk_per_lot: float
    requested_risk_amount: float
    maximum_risk_amount: float
    final_risk_amount: float
    margin_per_lot: float
    maximum_margin_amount: float
    final_margin_amount: float
    limiting_factors: tuple[str, ...] = ()
    rejection_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status is SizingStatus.ACCEPTED
