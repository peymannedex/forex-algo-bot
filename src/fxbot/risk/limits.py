"""Portfolio-level risk-limit configuration and violation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from math import isfinite


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _fraction(value: float, field_name: str, *, allow_one: bool = True) -> float:
    number = _positive(value, field_name)
    maximum = 1.0 if allow_one else 1.0 - 1e-15
    if number > maximum:
        raise ValueError(f"{field_name} cannot exceed 1")
    return number


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


class RiskLimitCode(StrEnum):
    """Stable machine-readable reason codes emitted by the portfolio guard."""

    MANUAL_KILL_SWITCH = "manual_kill_switch"
    AUTOMATIC_KILL_SWITCH = "automatic_kill_switch"
    EQUITY_STOP = "equity_stop"
    DAILY_REALIZED_LOSS = "daily_realized_loss"
    DAILY_TOTAL_LOSS = "daily_total_loss"
    INTRADAY_DRAWDOWN = "intraday_drawdown"
    CONSECUTIVE_LOSS_COOLDOWN = "consecutive_loss_cooldown"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_PENDING_ORDERS = "max_pending_orders"
    MAX_POSITIONS_PER_SYMBOL = "max_positions_per_symbol"
    MAX_TOTAL_RISK = "max_total_risk"
    MAX_GROSS_NOTIONAL = "max_gross_notional"
    MAX_NET_NOTIONAL = "max_net_notional"
    MAX_MARGIN_UTILIZATION = "max_margin_utilization"
    MAX_CURRENCY_CONCENTRATION = "max_currency_concentration"
    PROTECTIVE_STOP_REQUIRED = "protective_stop_required"


@dataclass(frozen=True, slots=True)
class PortfolioRiskLimits:
    """Hard portfolio and intraday controls applied to every proposed trade."""

    max_open_positions: int = 10
    max_pending_orders: int = 10
    max_positions_per_symbol: int = 3
    max_total_risk_fraction: float = 0.06
    max_gross_notional_multiple: float = 20.0
    max_net_notional_multiple: float = 10.0
    max_margin_utilization: float = 0.70
    max_currency_concentration_multiple: float = 10.0
    max_daily_realized_loss_fraction: float = 0.03
    max_daily_total_loss_fraction: float = 0.04
    max_intraday_drawdown_fraction: float = 0.05
    min_equity_fraction_of_balance: float = 0.75
    max_consecutive_losses: int = 4
    consecutive_loss_cooldown: timedelta = timedelta(minutes=30)
    include_pending_order_exposure: bool = True
    require_protective_stop: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_open_positions",
            _positive_int(self.max_open_positions, "max_open_positions"),
        )
        object.__setattr__(
            self,
            "max_pending_orders",
            _positive_int(self.max_pending_orders, "max_pending_orders"),
        )
        object.__setattr__(
            self,
            "max_positions_per_symbol",
            _positive_int(self.max_positions_per_symbol, "max_positions_per_symbol"),
        )
        object.__setattr__(
            self,
            "max_total_risk_fraction",
            _fraction(self.max_total_risk_fraction, "max_total_risk_fraction"),
        )
        object.__setattr__(
            self,
            "max_gross_notional_multiple",
            _positive(self.max_gross_notional_multiple, "max_gross_notional_multiple"),
        )
        object.__setattr__(
            self,
            "max_net_notional_multiple",
            _positive(self.max_net_notional_multiple, "max_net_notional_multiple"),
        )
        object.__setattr__(
            self,
            "max_margin_utilization",
            _fraction(self.max_margin_utilization, "max_margin_utilization"),
        )
        object.__setattr__(
            self,
            "max_currency_concentration_multiple",
            _positive(
                self.max_currency_concentration_multiple,
                "max_currency_concentration_multiple",
            ),
        )
        object.__setattr__(
            self,
            "max_daily_realized_loss_fraction",
            _fraction(
                self.max_daily_realized_loss_fraction,
                "max_daily_realized_loss_fraction",
            ),
        )
        object.__setattr__(
            self,
            "max_daily_total_loss_fraction",
            _fraction(self.max_daily_total_loss_fraction, "max_daily_total_loss_fraction"),
        )
        object.__setattr__(
            self,
            "max_intraday_drawdown_fraction",
            _fraction(
                self.max_intraday_drawdown_fraction,
                "max_intraday_drawdown_fraction",
            ),
        )
        object.__setattr__(
            self,
            "min_equity_fraction_of_balance",
            _fraction(
                self.min_equity_fraction_of_balance,
                "min_equity_fraction_of_balance",
            ),
        )
        object.__setattr__(
            self,
            "max_consecutive_losses",
            _positive_int(self.max_consecutive_losses, "max_consecutive_losses"),
        )
        if self.consecutive_loss_cooldown <= timedelta(0):
            raise ValueError("consecutive_loss_cooldown must be positive")


@dataclass(frozen=True, slots=True)
class RiskViolation:
    """One breached risk limit with observed and configured values."""

    code: RiskLimitCode
    message: str
    observed: float | int | str | None = None
    limit: float | int | str | None = None
    scope: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", RiskLimitCode(self.code))
        message = self.message.strip()
        if not message:
            raise ValueError("message cannot be empty")
        object.__setattr__(self, "message", message)
        if self.scope is not None:
            scope = self.scope.strip()
            object.__setattr__(self, "scope", scope or None)
