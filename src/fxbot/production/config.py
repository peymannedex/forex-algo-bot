"""Environment-driven production configuration with explicit live-trading gates."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Self

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeploymentProfile(StrEnum):
    """Supported runtime profiles."""

    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


class ProductionSettings(BaseSettings):
    """Validated runtime settings loaded from ``FXBOT_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FXBOT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    profile: DeploymentProfile = DeploymentProfile.PAPER
    service_name: str = "forex-algo-bot"
    symbols: tuple[str, ...] = ("EURUSD",)
    state_directory: Path = Path("var/state")
    log_directory: Path = Path("var/log")

    heartbeat_interval_seconds: float = 5.0
    reconciliation_interval_seconds: float = 30.0
    health_stale_after_seconds: float = 20.0
    max_quote_age_seconds: float = 3.0
    max_spread_bps: float = 12.0
    max_daily_loss: float = 500.0
    max_drawdown: float = 1_000.0

    auto_cancel_on_trip: bool = True
    auto_flatten_on_trip: bool = False
    demo_order_submission_enabled: bool = False
    live_trading_enabled: bool = False
    live_confirmation: SecretStr | None = None

    mt5_terminal_path: Path | None = None
    mt5_login: int | None = None
    mt5_password: SecretStr | None = None
    mt5_server: str | None = None
    mt5_magic_number: int = 51001
    mt5_deviation_points: int = 20

    @field_validator("service_name")
    @classmethod
    def _service_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("service_name cannot be empty")
        return normalized

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(item.strip().upper() for item in value if item.strip()))
        if not normalized:
            raise ValueError("symbols cannot be empty")
        return normalized

    @field_validator(
        "heartbeat_interval_seconds",
        "reconciliation_interval_seconds",
        "health_stale_after_seconds",
        "max_quote_age_seconds",
        "max_spread_bps",
        "max_daily_loss",
        "max_drawdown",
    )
    @classmethod
    def _positive_finite(cls, value: float) -> float:
        number = float(value)
        if not isfinite(number) or number <= 0.0:
            raise ValueError("value must be a positive finite number")
        return number

    @field_validator("mt5_magic_number")
    @classmethod
    def _positive_magic(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("mt5_magic_number must be positive")
        return value

    @field_validator("mt5_deviation_points")
    @classmethod
    def _non_negative_deviation(cls, value: int) -> int:
        if value < 0:
            raise ValueError("mt5_deviation_points must be non-negative")
        return value

    @model_validator(mode="after")
    def _live_gate(self) -> Self:
        if self.profile is DeploymentProfile.LIVE:
            confirmation = (
                self.live_confirmation.get_secret_value()
                if self.live_confirmation is not None
                else ""
            )
            if not self.live_trading_enabled:
                raise ValueError("LIVE profile requires live_trading_enabled=true")
            if confirmation != "I_UNDERSTAND_LIVE_TRADING":
                raise ValueError(
                    "LIVE profile requires FXBOT_LIVE_CONFIRMATION="
                    "I_UNDERSTAND_LIVE_TRADING"
                )
            if self.mt5_login is None or self.mt5_password is None or not self.mt5_server:
                raise ValueError("LIVE profile requires MT5 login, password, and server")
        return self

    @property
    def broker_dry_run(self) -> bool:
        """Return whether broker submission must remain in validation-only mode."""

        if self.profile is DeploymentProfile.LIVE:
            return False
        if self.profile is DeploymentProfile.DEMO:
            return not self.demo_order_submission_enabled
        return True

    def redacted(self) -> dict[str, object]:
        """Return a serializable settings view that never exposes secrets."""

        data = self.model_dump(mode="json")
        if self.mt5_password is not None:
            data["mt5_password"] = "***"
        if self.live_confirmation is not None:
            data["live_confirmation"] = "***"
        return data
