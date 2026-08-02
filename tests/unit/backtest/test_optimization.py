from __future__ import annotations

import pytest

from fxbot.backtest.optimization import (
    ObjectiveDirection,
    ObjectiveObservation,
    OptimizationConfig,
    SearchMethod,
    optimize,
)
from fxbot.backtest.parameters import IntegerParameter, ParameterSet, ParameterSpace


def space() -> ParameterSpace:
    return ParameterSpace((IntegerParameter("x", 1, 5),))


def score(parameters: ParameterSet) -> ObjectiveObservation:
    value = int(parameters["x"])
    return ObjectiveObservation(score=-abs(value - 4), trade_count=value)


def test_grid_optimization_finds_maximum() -> None:
    result = optimize(space(), score, OptimizationConfig())
    assert result.best is not None
    assert result.best.parameters["x"] == 4
    assert result.best.rank == 1


def test_minimization_direction() -> None:
    result = optimize(
        space(),
        lambda parameters: ObjectiveObservation(score=float(parameters["x"])),
        OptimizationConfig(direction=ObjectiveDirection.MINIMIZE),
    )
    assert result.best is not None
    assert result.best.parameters["x"] == 1


def test_minimum_trade_count_excludes_candidates() -> None:
    result = optimize(
        space(),
        score,
        OptimizationConfig(min_trade_count=5),
    )
    assert result.best is not None
    assert result.best.parameters["x"] == 5
    assert len(result.eligible_records) == 1


def test_random_search_is_seeded() -> None:
    config = OptimizationConfig(
        method=SearchMethod.RANDOM,
        max_evaluations=3,
        seed=11,
    )
    assert optimize(space(), score, config).fingerprint == optimize(
        space(), score, config
    ).fingerprint


def test_evaluator_errors_are_recorded() -> None:
    def evaluator(parameters: ParameterSet) -> ObjectiveObservation:
        if parameters["x"] == 3:
            raise RuntimeError("boom")
        return score(parameters)

    result = optimize(space(), evaluator, OptimizationConfig())
    failures = [item for item in result.records if item.error]
    assert len(failures) == 1
    assert "boom" in str(failures[0].error)


def test_fail_fast_propagates_evaluator_error() -> None:
    def evaluator(parameters: ParameterSet) -> ObjectiveObservation:
        raise RuntimeError(str(parameters["x"]))

    with pytest.raises(RuntimeError):
        optimize(space(), evaluator, OptimizationConfig(fail_fast=True))


def test_complexity_penalty_changes_ranking() -> None:
    def evaluator(parameters: ParameterSet) -> ObjectiveObservation:
        value = int(parameters["x"])
        return ObjectiveObservation(
            score=float(value),
            complexity=float(value),
        )

    result = optimize(
        space(),
        evaluator,
        OptimizationConfig(complexity_penalty=2.0),
    )
    assert result.best is not None
    assert result.best.parameters["x"] == 1
