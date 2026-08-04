"""Configuration for sustained paper-mode live market-data operation."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PaperLiveFeedSettings(BaseSettings):
    """Settings loaded from ``FXBOT_PAPER_LIVE_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="FXBOT_PAPER_LIVE_",
        case_sensitive=False,
        extra="ignore",
    )

    source: Literal["mt5"] = "mt5"
    poll_interval_seconds: float = 1.0
    reconnect_delay_seconds: float = 5.0
    max_consecutive_errors: int = 5
    history_bars_per_timeframe: int = 1_000
    evidence_directory: Path = Path("C:/forex-algo-bot/evidence/paper-soak")
    report_interval_seconds: float = 60.0
    stop_filename: str = "STOP_PAPER_SOAK"
    mt5_server_utc_offset_minutes: int = 0
    max_future_skew_seconds: float = 5.0

    @field_validator(
        "poll_interval_seconds",
        "reconnect_delay_seconds",
        "report_interval_seconds",
    )
    @classmethod
    def _positive_seconds(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or number <= 0.0:
            raise ValueError("time interval must be positive and finite")
        return number

    @field_validator("max_future_skew_seconds")
    @classmethod
    def _non_negative_seconds(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or number < 0.0:
            raise ValueError("future skew must be finite and non-negative")
        return number

    @field_validator(
        "max_consecutive_errors",
        "history_bars_per_timeframe",
    )
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be at least one")
        return value

    @field_validator("mt5_server_utc_offset_minutes")
    @classmethod
    def _bounded_server_offset(cls, value: int) -> int:
        if not -1_440 <= value <= 1_440:
            raise ValueError("MT5 server UTC offset must be within +/- 1440 minutes")
        return value

    @field_validator("stop_filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or Path(normalized).name != normalized:
            raise ValueError("stop_filename must be a plain file name")
        return normalized
