from __future__ import annotations

from datetime import timedelta

import pytest

from fxbot.risk.limits import PortfolioRiskLimits, RiskLimitCode, RiskViolation


def test_default_limits_are_production_safe_and_consistent() -> None:
    limits = PortfolioRiskLimits()
    assert limits.max_open_positions == 10
    assert limits.max_total_risk_fraction == 0.06
    assert limits.require_protective_stop
    assert limits.include_pending_order_exposure


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_open_positions", 0),
        ("max_pending_orders", 0),
        ("max_positions_per_symbol", 0),
        ("max_total_risk_fraction", 0),
        ("max_total_risk_fraction", 1.1),
        ("max_margin_utilization", 1.1),
        ("max_daily_total_loss_fraction", 0),
        ("min_equity_fraction_of_balance", 1.1),
        ("max_gross_notional_multiple", 0),
        ("max_consecutive_losses", 0),
    ],
)
def test_limits_reject_invalid_values(field: str, value: float | int) -> None:
    values: dict[str, object] = {}
    values[field] = value
    with pytest.raises(ValueError):
        PortfolioRiskLimits(**values)  # type: ignore[arg-type]


def test_limits_reject_non_positive_cooldown() -> None:
    with pytest.raises(ValueError, match="cooldown"):
        PortfolioRiskLimits(consecutive_loss_cooldown=timedelta(0))


def test_risk_violation_normalizes_code_and_scope() -> None:
    violation = RiskViolation(
        code="max_total_risk",
        message="  exceeded  ",
        observed=0.1,
        limit=0.06,
        scope=" EURUSD ",
    )
    assert violation.code is RiskLimitCode.MAX_TOTAL_RISK
    assert violation.message == "exceeded"
    assert violation.scope == "EURUSD"


def test_risk_violation_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="message"):
        RiskViolation(code=RiskLimitCode.EQUITY_STOP, message=" ")
