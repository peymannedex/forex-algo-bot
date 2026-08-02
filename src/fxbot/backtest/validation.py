"""Research validation gates for walk-forward and robustness results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite

from fxbot.backtest.robustness import (
    MultipleTestingAdjustment,
    PBOResult,
    SensitivityAnalysis,
)
from fxbot.backtest.walk_forward import WalkForwardResult


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    min_mean_train_score: float = 0.0
    min_mean_test_score: float = 0.0
    min_oos_efficiency: float = 0.0
    min_positive_test_fraction: float = 0.5
    max_parameter_switch_rate: float = 1.0
    min_total_test_trades: int = 0
    max_oos_drawdown_fraction: float = 1.0
    min_stable_neighbor_fraction: float = 0.0
    max_pbo: float = 1.0
    max_adjusted_p_value: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "min_mean_train_score",
            "min_mean_test_score",
            "min_oos_efficiency",
        ):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in (
            "min_positive_test_fraction",
            "max_parameter_switch_rate",
            "max_oos_drawdown_fraction",
            "min_stable_neighbor_fraction",
            "max_pbo",
            "max_adjusted_p_value",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_total_test_trades < 0:
            raise ValueError("min_total_test_trades must be non-negative")


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    passed: bool
    actual: float
    threshold: float
    comparator: str
    message: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if self.comparator not in {">=", "<="}:
            raise ValueError("comparator must be '>=' or '<='")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        if not self.checks:
            raise ValueError("checks cannot be empty")
        names = tuple(item.name for item in self.checks)
        if len(set(names)) != len(names):
            raise ValueError("check names must be unique")

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.checks)

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(item for item in self.checks if not item.passed)

    @property
    def fingerprint(self) -> str:
        payload = [
            [
                item.name,
                item.passed,
                item.actual,
                item.threshold,
                item.comparator,
                item.message,
            ]
            for item in self.checks
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def validate_research(
    walk_forward: WalkForwardResult,
    policy: ValidationPolicy,
    *,
    oos_drawdown_fraction: float,
    sensitivity: SensitivityAnalysis | None = None,
    pbo: PBOResult | None = None,
    multiple_testing: MultipleTestingAdjustment | None = None,
) -> ValidationReport:
    """Apply explicit pass/fail gates to one research experiment."""

    if not isfinite(oos_drawdown_fraction) or not 0.0 <= oos_drawdown_fraction <= 1.0:
        raise ValueError("oos_drawdown_fraction must be between 0 and 1")

    checks = [
        _minimum_check(
            "mean_train_score",
            walk_forward.mean_train_score,
            policy.min_mean_train_score,
        ),
        _minimum_check(
            "mean_test_score",
            walk_forward.mean_test_score,
            policy.min_mean_test_score,
        ),
        _minimum_check(
            "oos_efficiency",
            walk_forward.mean_efficiency,
            policy.min_oos_efficiency,
        ),
        _minimum_check(
            "positive_test_fraction",
            walk_forward.positive_test_fraction,
            policy.min_positive_test_fraction,
        ),
        _maximum_check(
            "parameter_switch_rate",
            walk_forward.parameter_switch_rate,
            policy.max_parameter_switch_rate,
        ),
        _minimum_check(
            "total_test_trades",
            float(walk_forward.total_test_trades),
            float(policy.min_total_test_trades),
        ),
        _maximum_check(
            "oos_drawdown_fraction",
            oos_drawdown_fraction,
            policy.max_oos_drawdown_fraction,
        ),
    ]
    if sensitivity is not None:
        checks.append(
            _minimum_check(
                "stable_neighbor_fraction",
                sensitivity.stable_fraction,
                policy.min_stable_neighbor_fraction,
            )
        )
    if pbo is not None:
        checks.append(
            _maximum_check(
                "probability_of_backtest_overfitting",
                pbo.probability,
                policy.max_pbo,
            )
        )
    if multiple_testing is not None:
        checks.append(
            _maximum_check(
                "sidak_adjusted_p_value",
                multiple_testing.sidak_p_value,
                policy.max_adjusted_p_value,
            )
        )
    return ValidationReport(tuple(checks))


def _minimum_check(name: str, actual: float, threshold: float) -> ValidationCheck:
    passed = actual >= threshold
    return ValidationCheck(
        name=name,
        passed=passed,
        actual=actual,
        threshold=threshold,
        comparator=">=",
        message=(
            f"{name}={actual:.6g} meets minimum {threshold:.6g}"
            if passed
            else f"{name}={actual:.6g} is below minimum {threshold:.6g}"
        ),
    )


def _maximum_check(name: str, actual: float, threshold: float) -> ValidationCheck:
    passed = actual <= threshold
    return ValidationCheck(
        name=name,
        passed=passed,
        actual=actual,
        threshold=threshold,
        comparator="<=",
        message=(
            f"{name}={actual:.6g} is within maximum {threshold:.6g}"
            if passed
            else f"{name}={actual:.6g} exceeds maximum {threshold:.6g}"
        ),
    )
