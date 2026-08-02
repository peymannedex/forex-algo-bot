from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.optimization import ObjectiveObservation, OptimizationConfig
from fxbot.backtest.parameters import IntegerParameter, ParameterSet, ParameterSpace
from fxbot.backtest.walk_forward import (
    WalkForwardConfig,
    WindowMode,
    generate_windows,
    run_walk_forward,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 7, 1, tzinfo=UTC)


def test_generate_rolling_windows() -> None:
    config = WalkForwardConfig(
        train_size=timedelta(days=60),
        test_size=timedelta(days=30),
        step_size=timedelta(days=30),
        purge_gap=timedelta(days=2),
    )
    windows = generate_windows(START, END, config)
    assert windows
    assert windows[0].train_start == START
    assert windows[0].purge_duration == timedelta(days=2)
    assert windows[1].train_start > windows[0].train_start


def test_generate_anchored_windows_keeps_train_start() -> None:
    config = WalkForwardConfig(
        train_size=timedelta(days=60),
        test_size=timedelta(days=30),
        step_size=timedelta(days=30),
        mode=WindowMode.ANCHORED,
        max_windows=3,
    )
    windows = generate_windows(START, END, config)
    assert len(windows) == 3
    assert {item.train_start for item in windows} == {START}
    assert windows[1].train_end > windows[0].train_end


def test_invalid_window_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        WalkForwardConfig(
            train_size=timedelta(0),
            test_size=timedelta(days=1),
            step_size=timedelta(days=1),
        )


def test_run_walk_forward_selects_training_best_without_lookahead() -> None:
    space = ParameterSpace((IntegerParameter("x", 1, 3),))
    config = WalkForwardConfig(
        train_size=timedelta(days=60),
        test_size=timedelta(days=30),
        step_size=timedelta(days=30),
        max_windows=2,
    )

    def evaluator(
        parameters: ParameterSet,
        start: datetime,
        end: datetime,
    ) -> ObjectiveObservation:
        del end
        x = int(parameters["x"])
        # Training windows begin before March; the future test score reverses.
        if start < datetime(2025, 3, 1, tzinfo=UTC):
            return ObjectiveObservation(score=float(x), trade_count=10)
        return ObjectiveObservation(score=float(4 - x), trade_count=8)

    result = run_walk_forward(
        space,
        evaluator,
        start=START,
        end=END,
        walk_forward_config=config,
        optimization_config=OptimizationConfig(),
    )
    assert len(result.folds) == 2
    assert result.folds[0].selected_parameters["x"] == 3
    assert result.total_test_trades == 16
    assert result.fingerprint


def test_run_walk_forward_rejects_empty_window_plan() -> None:
    space = ParameterSpace((IntegerParameter("x", 1, 2),))
    with pytest.raises(ValueError, match="no walk-forward windows"):
        run_walk_forward(
            space,
            lambda parameters, start, end: ObjectiveObservation(score=1.0),
            start=START,
            end=START + timedelta(days=10),
            walk_forward_config=WalkForwardConfig(
                train_size=timedelta(days=30),
                test_size=timedelta(days=10),
                step_size=timedelta(days=10),
            ),
            optimization_config=OptimizationConfig(),
        )
