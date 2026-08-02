"""Deterministic parameter search and objective ranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from typing import Protocol

from fxbot.backtest.parameters import ParameterSet, ParameterSpace


class SearchMethod(StrEnum):
    GRID = "grid"
    RANDOM = "random"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class ObjectiveObservation:
    """One evaluator response with an auditable primary score."""

    score: float
    trade_count: int = 0
    complexity: float = 0.0
    secondary: tuple[tuple[str, float], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        if not isfinite(self.complexity) or self.complexity < 0.0:
            raise ValueError("complexity must be finite and non-negative")
        secondary_names: set[str] = set()
        normalized_secondary: list[tuple[str, float]] = []
        for raw_name, raw_value in self.secondary:
            name = raw_name.strip()
            if not name or name in secondary_names:
                raise ValueError("secondary metric names must be non-empty and unique")
            value = float(raw_value)
            if not isfinite(value):
                raise ValueError("secondary metric values must be finite")
            secondary_names.add(name)
            normalized_secondary.append((name, value))
        object.__setattr__(self, "secondary", tuple(sorted(normalized_secondary)))
        normalized_metadata = tuple(
            sorted((name.strip(), value.strip()) for name, value in self.metadata)
        )
        if any(not name for name, _ in normalized_metadata):
            raise ValueError("metadata names cannot be empty")
        if len({name for name, _ in normalized_metadata}) != len(normalized_metadata):
            raise ValueError("metadata names must be unique")
        object.__setattr__(self, "metadata", normalized_metadata)

    def secondary_value(self, name: str, default: float = 0.0) -> float:
        return dict(self.secondary).get(name, default)


class ObjectiveEvaluator(Protocol):
    def __call__(self, parameters: ParameterSet) -> ObjectiveObservation: ...


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    method: SearchMethod = SearchMethod.GRID
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    max_evaluations: int | None = None
    seed: int = 0
    min_trade_count: int = 0
    complexity_penalty: float = 0.0
    fail_fast: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", SearchMethod(self.method))
        object.__setattr__(self, "direction", ObjectiveDirection(self.direction))
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.min_trade_count < 0:
            raise ValueError("min_trade_count must be non-negative")
        if not isfinite(self.complexity_penalty) or self.complexity_penalty < 0.0:
            raise ValueError("complexity_penalty must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    parameters: ParameterSet
    observation: ObjectiveObservation | None
    adjusted_score: float | None
    rank: int | None
    eligible: bool
    error: str | None = None

    def __post_init__(self) -> None:
        if self.observation is None:
            if self.adjusted_score is not None or self.eligible:
                raise ValueError("failed evaluations cannot be eligible or scored")
        elif self.adjusted_score is None:
            raise ValueError("successful evaluations require adjusted_score")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("rank must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    config: OptimizationConfig
    space_fingerprint: str
    records: tuple[EvaluationRecord, ...]

    @property
    def successful_records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(item for item in self.records if item.observation is not None)

    @property
    def eligible_records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(item for item in self.records if item.eligible)

    @property
    def best(self) -> EvaluationRecord | None:
        return next((item for item in self.records if item.rank == 1), None)

    @property
    def fingerprint(self) -> str:
        payload = {
            "config": {
                "method": self.config.method.value,
                "direction": self.config.direction.value,
                "max_evaluations": self.config.max_evaluations,
                "seed": self.config.seed,
                "min_trade_count": self.config.min_trade_count,
                "complexity_penalty": self.config.complexity_penalty,
            },
            "space": self.space_fingerprint,
            "records": [
                {
                    "parameters": item.parameters.as_dict(),
                    "score": (
                        None if item.observation is None else item.observation.score
                    ),
                    "adjusted_score": item.adjusted_score,
                    "rank": item.rank,
                    "eligible": item.eligible,
                    "error": item.error,
                }
                for item in self.records
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def optimize(
    space: ParameterSpace,
    evaluator: ObjectiveEvaluator,
    config: OptimizationConfig | None = None,
) -> OptimizationResult:
    """Evaluate a finite parameter space and deterministically rank candidates."""

    resolved_config = config or OptimizationConfig()
    if resolved_config.method is SearchMethod.GRID:
        candidates = space.grid(limit=resolved_config.max_evaluations)
    else:
        evaluations = resolved_config.max_evaluations or space.cardinality
        candidates = space.random_sample(evaluations, seed=resolved_config.seed)

    raw_records: list[EvaluationRecord] = []
    for parameters in candidates:
        try:
            observation = evaluator(parameters)
        except Exception as exc:
            if resolved_config.fail_fast:
                raise
            raw_records.append(
                EvaluationRecord(
                    parameters=parameters,
                    observation=None,
                    adjusted_score=None,
                    rank=None,
                    eligible=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        adjusted = _adjusted_score(observation, resolved_config)
        is_eligible = (
            observation.trade_count >= resolved_config.min_trade_count
        )
        raw_records.append(
            EvaluationRecord(
                parameters=parameters,
                observation=observation,
                adjusted_score=adjusted,
                rank=None,
                eligible=is_eligible,
            )
        )

    eligible_records = [item for item in raw_records if item.eligible]
    reverse = resolved_config.direction is ObjectiveDirection.MAXIMIZE
    eligible_records.sort(
        key=lambda item: (
            _record_score(item),
            item.parameters.fingerprint,
        ),
        reverse=reverse,
    )
    ranks = {item.parameters.fingerprint: index + 1 for index, item in enumerate(eligible_records)}
    ranked = tuple(
        replace(item, rank=ranks.get(item.parameters.fingerprint))
        for item in raw_records
    )
    return OptimizationResult(
        config=resolved_config,
        space_fingerprint=space.fingerprint,
        records=ranked,
    )


def observation_from_backtest(
    result: object,
    *,
    metric: str = "sharpe_ratio",
    complexity: float = 0.0,
) -> ObjectiveObservation:
    """Build an objective from an existing ``BacktestResult`` without hard coupling."""

    from fxbot.backtest.metrics import calculate_performance
    from fxbot.backtest.results import BacktestResult

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    performance = calculate_performance(result)
    trade_statistics = performance.trade_statistics
    metric_values = {
        "total_return": performance.total_return,
        "cagr": performance.cagr,
        "sharpe_ratio": performance.sharpe_ratio,
        "sortino_ratio": performance.sortino_ratio,
        "calmar_ratio": performance.calmar_ratio,
        "recovery_factor": performance.recovery_factor,
        "expectancy": trade_statistics.expectancy,
        "profit_factor": trade_statistics.profit_factor,
        "maximum_drawdown_fraction": performance.maximum_drawdown_fraction,
    }
    try:
        score = metric_values[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported objective metric: {metric}") from exc
    if not isfinite(score):
        score = 1e12 if score > 0.0 else -1e12
    return ObjectiveObservation(
        score=float(score),
        trade_count=trade_statistics.trade_count,
        complexity=complexity,
        secondary=(
            ("total_return", performance.total_return),
            ("maximum_drawdown_fraction", performance.maximum_drawdown_fraction),
        ),
        metadata=(("result_fingerprint", result.fingerprint),),
    )


def _adjusted_score(
    observation: ObjectiveObservation,
    config: OptimizationConfig,
) -> float:
    penalty = config.complexity_penalty * observation.complexity
    if config.direction is ObjectiveDirection.MAXIMIZE:
        return observation.score - penalty
    return observation.score + penalty


def _record_score(record: EvaluationRecord) -> float:
    if record.adjusted_score is None:
        raise ValueError("eligible record is missing adjusted_score")
    return record.adjusted_score
