"""Environment settings for deterministic paper integration and replay."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fxbot.domain.enums import Timeframe


class PaperIntegrationSettings(BaseSettings):
    """Settings loaded from ``FXBOT_PAPER_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FXBOT_PAPER_",
        case_sensitive=False,
        extra="ignore",
    )

    initial_balance: float = 100_000.0
    account_currency: str = "USD"
    leverage: float = 100.0
    risk_fraction: float = 0.005
    fixed_quantity: float | None = None

    primary_timeframe: str = "M5"
    required_timeframes: str = "M5,M15"
    warmup_bars: int = 50
    min_confidence: float = 0.55
    duplicate_suppression_seconds: float = 300.0

    max_abs_position_per_symbol: float = 1.0
    max_gross_quantity: float = 3.0
    commission_per_unit: float = 0.0
    slippage: float = 0.0
    max_fill_quantity_per_quote: float | None = None

    replay_csv: Path = Path("data/paper/paper_replay.csv")
    state_filename: str = "paper_runtime_state.json"
    journal_filename: str = "paper_execution_journal.json"

    @field_validator("account_currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("account_currency cannot be empty")
        return normalized

    @field_validator("initial_balance", "leverage")
    @classmethod
    def _positive(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or number <= 0.0:
            raise ValueError("value must be positive and finite")
        return number

    @field_validator(
        "risk_fraction",
        "min_confidence",
    )
    @classmethod
    def _fraction(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or not 0.0 < number <= 1.0:
            raise ValueError("value must be between zero and one")
        return number

    @field_validator(
        "duplicate_suppression_seconds",
        "commission_per_unit",
        "slippage",
    )
    @classmethod
    def _non_negative(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or number < 0.0:
            raise ValueError("value must be finite and non-negative")
        return number

    @field_validator(
        "fixed_quantity",
        "max_fill_quantity_per_quote",
    )
    @classmethod
    def _optional_positive(cls, value: float | None) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not isfinite(number) or number <= 0.0:
            raise ValueError("value must be positive and finite")
        return number

    @field_validator(
        "max_abs_position_per_symbol",
        "max_gross_quantity",
    )
    @classmethod
    def _positive_limit(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or number <= 0.0:
            raise ValueError("exposure limit must be positive and finite")
        return number

    @field_validator("warmup_bars")
    @classmethod
    def _warmup(cls, value: int) -> int:
        if value < 2:
            raise ValueError("warmup_bars must be at least two")
        return value

    @property
    def parsed_primary_timeframe(self) -> Timeframe:
        parsed = Timeframe.parse(self.primary_timeframe)
        if parsed is Timeframe.TICK:
            raise ValueError("primary_timeframe cannot be tick")
        return parsed

    @property
    def parsed_required_timeframes(self) -> tuple[Timeframe, ...]:
        values = tuple(
            Timeframe.parse(item.strip())
            for item in self.required_timeframes.split(",")
            if item.strip()
        )
        if not values:
            raise ValueError("required_timeframes cannot be empty")
        primary = self.parsed_primary_timeframe
        ordered = [primary]
        for timeframe in values:
            if timeframe is Timeframe.TICK:
                raise ValueError("required_timeframes cannot include tick")
            if timeframe not in ordered:
                ordered.append(timeframe)
        return tuple(ordered)
