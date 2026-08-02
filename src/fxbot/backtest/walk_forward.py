"""Rolling and anchored walk-forward optimization."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from statistics import fmean
from typing import Protocol

from fxbot.backtest.optimization import (
    ObjectiveObservation,
    OptimizationConfig,
    OptimizationResult,
    optimize,
)
from fxbot.backtest.parameters import ParameterSet, ParameterSpace


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class WindowMode(StrEnum):
    ROLLING = "rolling"
    ANCHORED = "anchored"


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_size: timedelta
    test_size: timedelta
    step_size: timedelta
    mode: WindowMode = WindowMode.ROLLING
    purge_gap: timedelta = field(default_factory=timedelta)
    max_windows: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", WindowMode(self.mode))
        if self.train_size <= timedelta(0):
            raise ValueError("train_size must be positive")
        if self.test_size <= timedelta(0):
            raise ValueError("test_size must be positive")
        if self.step_size <= timedelta(0):
            raise ValueError("step_size must be positive")
        if self.purge_gap < timedelta(0):
            raise ValueError("purge_gap cannot be negative")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive")


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be non-negative")
        for name in ("train_start", "train_end", "test_start", "test_end"):
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if not self.train_start < self.train_end <= self.test_start < self.test_end:
            raise ValueError("window timestamps must be ordered and non-overlapping")

    @property
    def purge_duration(self) -> timedelta:
        return self.test_start - self.train_end


class WindowEvaluator(Protocol):
    def __call__(
        self,
        parameters: ParameterSet,
        start: datetime,
        end: datetime,
    ) -> ObjectiveObservation: ...


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    window: WalkForwardWindow
    optimization: OptimizationResult
    selected_parameters: ParameterSet
    train_observation: ObjectiveObservation
    test_observation: ObjectiveObservation
    efficiency: float

    def __post_init__(self) -> None:
        if self.optimization.best is None:
            raise ValueError("optimization must contain an eligible best record")


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    config: WalkForwardConfig
    optimization_config: OptimizationConfig
    start: datetime
    end: datetime
    folds: tuple[WalkForwardFold, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _utc(self.start, "start"))
        object.__setattr__(self, "end", _utc(self.end, "end"))
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if not self.folds:
            raise ValueError("folds cannot be empty")

    @property
    def mean_train_score(self) -> float:
        return fmean(item.train_observation.score for item in self.folds)

    @property
    def mean_test_score(self) -> float:
        return fmean(item.test_observation.score for item in self.folds)

    @property
    def mean_efficiency(self) -> float:
        return fmean(item.efficiency for item in self.folds)

    @property
    def positive_test_fraction(self) -> float:
        return sum(item.test_observation.score > 0.0 for item in self.folds) / len(
            self.folds
        )

    @property
    def parameter_switch_rate(self) -> float:
        if len(self.folds) < 2:
            return 0.0
        switches = sum(
            current.selected_parameters != previous.selected_parameters
            for previous, current in itertools.pairwise(self.folds)
        )
        return switches / (len(self.folds) - 1)

    @property
    def total_test_trades(self) -> int:
        return sum(item.test_observation.trade_count for item in self.folds)

    @property
    def fingerprint(self) -> str:
        payload = {
            "period": [self.start.isoformat(), self.end.isoformat()],
            "mode": self.config.mode.value,
            "folds": [
                {
                    "window": [
                        item.window.train_start.isoformat(),
                        item.window.train_end.isoformat(),
                        item.window.test_start.isoformat(),
                        item.window.test_end.isoformat(),
                    ],
                    "parameters": item.selected_parameters.as_dict(),
                    "train": item.train_observation.score,
                    "test": item.test_observation.score,
                    "efficiency": item.efficiency,
                    "optimization": item.optimization.fingerprint,
                }
                for item in self.folds
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def generate_windows(
    start: datetime,
    end: datetime,
    config: WalkForwardConfig,
) -> tuple[WalkForwardWindow, ...]:
    """Generate half-open train/test windows without overlap."""

    period_start = _utc(start, "start")
    period_end = _utc(end, "end")
    if period_end <= period_start:
        raise ValueError("end must be after start")
    windows: list[WalkForwardWindow] = []
    first_test_start = period_start + config.train_size + config.purge_gap
    test_start = first_test_start
    index = 0
    while test_start + config.test_size <= period_end:
        train_end = test_start - config.purge_gap
        train_start = (
            period_start
            if config.mode is WindowMode.ANCHORED
            else train_end - config.train_size
        )
        windows.append(
            WalkForwardWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_start + config.test_size,
            )
        )
        if config.max_windows is not None and len(windows) >= config.max_windows:
            break
        index += 1
        test_start = first_test_start + index * config.step_size
    return tuple(windows)


def run_walk_forward(
    space: ParameterSpace,
    evaluator: WindowEvaluator,
    *,
    start: datetime,
    end: datetime,
    walk_forward_config: WalkForwardConfig,
    optimization_config: OptimizationConfig,
) -> WalkForwardResult:
    """Optimize on each training window and evaluate only on its future test window."""

    windows = generate_windows(start, end, walk_forward_config)
    if not windows:
        raise ValueError("configuration produced no walk-forward windows")
    folds: list[WalkForwardFold] = []
    for window in windows:
        fold_config = replace(
            optimization_config,
            seed=optimization_config.seed + window.index,
        )

        training_objective = partial(
            _evaluate_window,
            evaluator,
            window.train_start,
            window.train_end,
        )
        optimization = optimize(space, training_objective, fold_config)
        best = optimization.best
        if best is None or best.observation is None:
            raise ValueError(f"No eligible parameters for fold {window.index}")
        test_observation = evaluator(
            best.parameters,
            window.test_start,
            window.test_end,
        )
        folds.append(
            WalkForwardFold(
                window=window,
                optimization=optimization,
                selected_parameters=best.parameters,
                train_observation=best.observation,
                test_observation=test_observation,
                efficiency=_efficiency(
                    best.observation.score,
                    test_observation.score,
                ),
            )
        )
    return WalkForwardResult(
        config=walk_forward_config,
        optimization_config=optimization_config,
        start=start,
        end=end,
        folds=tuple(folds),
    )


def _efficiency(train_score: float, test_score: float) -> float:
    if train_score == 0.0:
        return 0.0
    return test_score / abs(train_score)


def _evaluate_window(
    evaluator: WindowEvaluator,
    start: datetime,
    end: datetime,
    parameters: ParameterSet,
) -> ObjectiveObservation:
    return evaluator(parameters, start, end)
