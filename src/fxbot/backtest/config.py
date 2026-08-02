"""Validated configuration models for deterministic backtest execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


def _currency(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("account_currency cannot be empty")
    return normalized


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class InstrumentConfig:
    """Backtest contract and margin settings for one symbol."""

    symbol: str
    contract_size: float = 100_000.0
    leverage: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(
            self,
            "contract_size",
            _positive(self.contract_size, "contract_size"),
        )
        leverage = _positive(self.leverage, "leverage")
        if leverage < 1.0:
            raise ValueError("leverage must be at least 1")
        object.__setattr__(self, "leverage", leverage)


@dataclass(frozen=True, slots=True)
class CommissionConfig:
    """Simple commission model charged once per simulated fill."""

    per_lot: float = 0.0
    minimum_per_fill: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_lot", _non_negative(self.per_lot, "per_lot"))
        object.__setattr__(
            self,
            "minimum_per_fill",
            _non_negative(self.minimum_per_fill, "minimum_per_fill"),
        )

    def calculate(self, volume: float) -> float:
        lots = _positive(volume, "volume")
        raw = lots * self.per_lot
        return max(raw, self.minimum_per_fill if raw > 0.0 else 0.0)


@dataclass(frozen=True, slots=True)
class SlippageConfig:
    """Adverse slippage with optional reproducible random jitter."""

    base_bps: float = 0.0
    jitter_bps: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_bps", _non_negative(self.base_bps, "base_bps"))
        object.__setattr__(self, "jitter_bps", _non_negative(self.jitter_bps, "jitter_bps"))


@dataclass(frozen=True, slots=True)
class SwapConfig:
    """Daily account-currency swap charged per lot at rollover."""

    long_per_lot: float = 0.0
    short_per_lot: float = 0.0
    rollover_hour_utc: int = 21
    triple_swap_weekday: int = 2

    def __post_init__(self) -> None:
        for name in ("long_per_lot", "short_per_lot"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0 <= self.rollover_hour_utc <= 23:
            raise ValueError("rollover_hour_utc must be between 0 and 23")
        if not 0 <= self.triple_swap_weekday <= 6:
            raise ValueError("triple_swap_weekday must be between 0 and 6")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Broker-simulation limits and rejection controls."""

    max_spread_bps: float | None = None
    max_fill_volume_per_event: float | None = None
    rejection_probability: float = 0.0
    slippage: SlippageConfig = field(default_factory=SlippageConfig)

    def __post_init__(self) -> None:
        if self.max_spread_bps is not None:
            object.__setattr__(
                self,
                "max_spread_bps",
                _positive(self.max_spread_bps, "max_spread_bps"),
            )
        if self.max_fill_volume_per_event is not None:
            object.__setattr__(
                self,
                "max_fill_volume_per_event",
                _positive(
                    self.max_fill_volume_per_event,
                    "max_fill_volume_per_event",
                ),
            )
        probability = float(self.rejection_probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("rejection_probability must be between 0 and 1")
        object.__setattr__(self, "rejection_probability", probability)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Top-level reproducibility and account settings."""

    initial_cash: float
    instruments: tuple[InstrumentConfig, ...]
    account_currency: str = "USD"
    seed: int = 7
    strict_input_order: bool = False
    liquidate_at_end: bool = True
    allow_short: bool = True
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    swap: SwapConfig = field(default_factory=SwapConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_cash", _positive(self.initial_cash, "initial_cash"))
        object.__setattr__(self, "account_currency", _currency(self.account_currency))
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.instruments:
            raise ValueError("At least one instrument is required")
        symbols = [item.symbol for item in self.instruments]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Instrument symbols must be unique")

    def instrument(self, symbol: str) -> InstrumentConfig:
        normalized = _symbol(symbol)
        for item in self.instruments:
            if item.symbol == normalized:
                return item
        raise KeyError(f"No backtest instrument configured for {normalized}")
