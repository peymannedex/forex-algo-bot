from __future__ import annotations

import pytest

from fxbot.backtest.optimization import ObjectiveObservation
from fxbot.backtest.parameters import IntegerParameter, ParameterSet, ParameterSpace
from fxbot.backtest.robustness import (
    adjust_for_multiple_testing,
    analyze_parameter_sensitivity,
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
)


def test_sensitivity_scores_neighbor_stability() -> None:
    space = ParameterSpace((IntegerParameter("x", 1, 5),))
    center = ParameterSet.from_mapping({"x": 3})

    def evaluator(parameters: ParameterSet) -> ObjectiveObservation:
        x = int(parameters["x"])
        return ObjectiveObservation(score=10.0 - abs(x - 3))

    result = analyze_parameter_sensitivity(
        space,
        center,
        evaluator,
        tolerance=0.11,
    )
    assert len(result.points) == 2
    assert result.stable_fraction == 1.0
    assert result.fingerprint


def test_multiple_testing_adjustments_are_bounded() -> None:
    result = adjust_for_multiple_testing(0.02, 10)
    assert result.bonferroni_p_value == pytest.approx(0.2)
    assert 0.0 < result.sidak_p_value < 1.0


def test_multiple_testing_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        adjust_for_multiple_testing(1.1, 2)


def test_pbo_detects_unstable_selection() -> None:
    matrix = (
        (10.0, 1.0, 2.0),
        (9.0, 2.0, 3.0),
        (1.0, 10.0, 2.0),
        (2.0, 9.0, 3.0),
        (8.0, 1.0, 4.0),
        (1.0, 8.0, 4.0),
    )
    result = probability_of_backtest_overfitting(matrix)
    assert 0.0 <= result.probability <= 1.0
    assert result.combinations_evaluated > 0
    assert len(result.logits) == result.combinations_evaluated


def test_pbo_rejects_odd_rows() -> None:
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(((1.0, 2.0),) * 5)


def test_deflated_sharpe_probability_is_bounded() -> None:
    result = deflated_sharpe_probability(
        1.5,
        trials=20,
        sample_count=250,
    )
    assert 0.0 <= result.probability <= 1.0
    assert result.standard_error > 0.0


def test_deflated_sharpe_rejects_small_sample() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe_probability(1.0, trials=2, sample_count=1)
