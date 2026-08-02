from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fxbot.backtest.optimization import ObjectiveObservation, OptimizationConfig
from fxbot.backtest.parameters import IntegerParameter, ParameterSet, ParameterSpace
from fxbot.backtest.robustness import (
    adjust_for_multiple_testing,
    analyze_parameter_sensitivity,
    probability_of_backtest_overfitting,
)
from fxbot.backtest.validation import ValidationPolicy, validate_research
from fxbot.backtest.walk_forward import WalkForwardConfig, run_walk_forward


def make_walk_forward():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 6, 1, tzinfo=UTC)
    space = ParameterSpace((IntegerParameter("x", 1, 3),))

    def evaluator(
        parameters: ParameterSet,
        window_start: datetime,
        window_end: datetime,
    ) -> ObjectiveObservation:
        del window_start, window_end
        value = int(parameters["x"])
        return ObjectiveObservation(
            score=float(value),
            trade_count=5,
        )

    return run_walk_forward(
        space,
        evaluator,
        start=start,
        end=end,
        walk_forward_config=WalkForwardConfig(
            train_size=timedelta(days=45),
            test_size=timedelta(days=20),
            step_size=timedelta(days=20),
            max_windows=3,
        ),
        optimization_config=OptimizationConfig(),
    )


def test_validation_report_passes_when_all_gates_are_met() -> None:
    walk_forward = make_walk_forward()
    report = validate_research(
        walk_forward,
        ValidationPolicy(
            min_mean_train_score=2.0,
            min_mean_test_score=2.0,
            min_oos_efficiency=0.8,
            min_positive_test_fraction=1.0,
            max_parameter_switch_rate=0.0,
            min_total_test_trades=15,
            max_oos_drawdown_fraction=0.2,
        ),
        oos_drawdown_fraction=0.1,
    )
    assert report.passed
    assert not report.failed_checks
    assert report.fingerprint


def test_validation_report_lists_failed_checks() -> None:
    walk_forward = make_walk_forward()
    report = validate_research(
        walk_forward,
        ValidationPolicy(
            min_mean_test_score=10.0,
            min_total_test_trades=100,
            max_oos_drawdown_fraction=0.01,
        ),
        oos_drawdown_fraction=0.5,
    )
    assert not report.passed
    assert {item.name for item in report.failed_checks} == {
        "mean_test_score",
        "total_test_trades",
        "oos_drawdown_fraction",
    }


def test_validation_includes_optional_robustness_gates() -> None:
    walk_forward = make_walk_forward()
    space = ParameterSpace((IntegerParameter("x", 1, 3),))
    center = ParameterSet.from_mapping({"x": 2})
    sensitivity = analyze_parameter_sensitivity(
        space,
        center,
        lambda parameters: ObjectiveObservation(
            score=10.0 - abs(int(parameters["x"]) - 2)
        ),
    )
    pbo = probability_of_backtest_overfitting(
        (
            (3.0, 1.0),
            (2.5, 1.2),
            (1.0, 3.0),
            (1.2, 2.5),
        )
    )
    adjustment = adjust_for_multiple_testing(0.01, 5)
    report = validate_research(
        walk_forward,
        ValidationPolicy(
            min_stable_neighbor_fraction=0.5,
            max_pbo=1.0,
            max_adjusted_p_value=0.1,
        ),
        oos_drawdown_fraction=0.1,
        sensitivity=sensitivity,
        pbo=pbo,
        multiple_testing=adjustment,
    )
    names = {item.name for item in report.checks}
    assert "stable_neighbor_fraction" in names
    assert "probability_of_backtest_overfitting" in names
    assert "sidak_adjusted_p_value" in names
