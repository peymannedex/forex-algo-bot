"""Parameter stability, multiple-testing, and overfitting diagnostics."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import erf, isfinite, log, sqrt
from statistics import fmean, median

from fxbot.backtest.optimization import ObjectiveDirection, ObjectiveObservation
from fxbot.backtest.parameters import ParameterSet, ParameterSpace


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    parameters: ParameterSet
    score: float
    relative_degradation: float
    stable: bool

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if not isfinite(self.relative_degradation):
            raise ValueError("relative_degradation must be finite")


@dataclass(frozen=True, slots=True)
class SensitivityAnalysis:
    center: ParameterSet
    baseline_score: float
    points: tuple[SensitivityPoint, ...]
    tolerance: float

    @property
    def stable_fraction(self) -> float:
        if not self.points:
            return 1.0
        return sum(item.stable for item in self.points) / len(self.points)

    @property
    def median_relative_degradation(self) -> float:
        return median(
            (item.relative_degradation for item in self.points),
        ) if self.points else 0.0

    @property
    def worst_relative_degradation(self) -> float:
        return max(
            (item.relative_degradation for item in self.points),
            default=0.0,
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "center": self.center.as_dict(),
            "baseline": self.baseline_score,
            "tolerance": self.tolerance,
            "points": [
                [
                    item.parameters.as_dict(),
                    item.score,
                    item.relative_degradation,
                    item.stable,
                ]
                for item in self.points
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MultipleTestingAdjustment:
    trials: int
    raw_p_value: float
    bonferroni_p_value: float
    sidak_p_value: float

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        for name in ("raw_p_value", "bonferroni_p_value", "sidak_p_value"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PBOResult:
    probability: float
    logits: tuple[float, ...]
    combinations_evaluated: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if self.combinations_evaluated != len(self.logits):
            raise ValueError("combinations_evaluated must match logits")

    @property
    def median_logit(self) -> float:
        return median(self.logits) if self.logits else 0.0


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    expected_maximum_sharpe: float
    standard_error: float
    probability: float


def analyze_parameter_sensitivity(
    space: ParameterSpace,
    center: ParameterSet,
    evaluator: Callable[[ParameterSet], ObjectiveObservation],
    *,
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
    tolerance: float = 0.10,
) -> SensitivityAnalysis:
    """Evaluate one-step neighbors around an optimized parameter set."""

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    space.validate(center)
    baseline = evaluator(center).score
    points: list[SensitivityPoint] = []
    for parameters in space.neighbors(center):
        score = evaluator(parameters).score
        degradation = _relative_degradation(
            baseline,
            score,
            direction=direction,
        )
        points.append(
            SensitivityPoint(
                parameters=parameters,
                score=score,
                relative_degradation=degradation,
                stable=degradation <= tolerance,
            )
        )
    return SensitivityAnalysis(
        center=center,
        baseline_score=baseline,
        points=tuple(points),
        tolerance=tolerance,
    )


def adjust_for_multiple_testing(
    raw_p_value: float,
    trials: int,
) -> MultipleTestingAdjustment:
    """Return conservative Bonferroni and Šidák family-wise adjustments."""

    if not isfinite(raw_p_value) or not 0.0 <= raw_p_value <= 1.0:
        raise ValueError("raw_p_value must be between 0 and 1")
    if trials <= 0:
        raise ValueError("trials must be positive")
    return MultipleTestingAdjustment(
        trials=trials,
        raw_p_value=raw_p_value,
        bonferroni_p_value=min(raw_p_value * trials, 1.0),
        sidak_p_value=min(1.0 - (1.0 - raw_p_value) ** trials, 1.0),
    )


def probability_of_backtest_overfitting(
    score_matrix: Sequence[Sequence[float]],
    *,
    max_combinations: int | None = 10_000,
) -> PBOResult:
    """Estimate PBO using combinatorially symmetric cross-validation.

    Rows are independent chronological subperiods and columns are strategy or
    parameter candidates. Each symmetric split selects the best in-sample
    candidate, measures its out-of-sample relative rank, and records a logit.
    """

    matrix = _validate_matrix(score_matrix)
    row_count = len(matrix)
    column_count = len(matrix[0])
    if row_count < 4 or row_count % 2 != 0:
        raise ValueError("score_matrix must contain an even number of at least four rows")
    if column_count < 2:
        raise ValueError("score_matrix must contain at least two candidates")
    if max_combinations is not None and max_combinations <= 0:
        raise ValueError("max_combinations must be positive")

    half = row_count // 2
    all_indices = tuple(range(row_count))
    logits: list[float] = []
    for train_indices in itertools.combinations(all_indices, half):
        if 0 not in train_indices:
            continue
        if max_combinations is not None and len(logits) >= max_combinations:
            break
        train_set = set(train_indices)
        test_indices = tuple(index for index in all_indices if index not in train_set)
        train_means = tuple(
            fmean(matrix[row][column] for row in train_indices)
            for column in range(column_count)
        )
        selected = max(
            range(column_count),
            key=lambda column: (train_means[column], -column),
        )
        test_means = tuple(
            fmean(matrix[row][column] for row in test_indices)
            for column in range(column_count)
        )
        sorted_columns = sorted(
            range(column_count),
            key=lambda column: (test_means[column], -column),
        )
        rank_from_worst = sorted_columns.index(selected) + 1
        relative_rank = rank_from_worst / (column_count + 1.0)
        logits.append(log(relative_rank / (1.0 - relative_rank)))

    if not logits:
        raise ValueError("No symmetric combinations were evaluated")
    probability = sum(value <= 0.0 for value in logits) / len(logits)
    return PBOResult(
        probability=probability,
        logits=tuple(logits),
        combinations_evaluated=len(logits),
    )


def deflated_sharpe_probability(
    observed_sharpe: float,
    *,
    trials: int,
    sample_count: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> DeflatedSharpeResult:
    """Approximate the probability that Sharpe exceeds selection bias."""

    for name, value in (
        ("observed_sharpe", observed_sharpe),
        ("skewness", skewness),
        ("excess_kurtosis", excess_kurtosis),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    expected_maximum = _expected_maximum_normal(trials)
    variance_term = (
        1.0
        - skewness * observed_sharpe
        + ((excess_kurtosis + 2.0) / 4.0) * observed_sharpe**2
    )
    standard_error = sqrt(max(variance_term / (sample_count - 1), 1e-18))
    z_score = (observed_sharpe - expected_maximum) / standard_error
    probability = _normal_cdf(z_score)
    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        expected_maximum_sharpe=expected_maximum,
        standard_error=standard_error,
        probability=probability,
    )


def _relative_degradation(
    baseline: float,
    score: float,
    *,
    direction: ObjectiveDirection,
) -> float:
    denominator = max(abs(baseline), 1e-12)
    if direction is ObjectiveDirection.MAXIMIZE:
        return max((baseline - score) / denominator, 0.0)
    return max((score - baseline) / denominator, 0.0)


def _validate_matrix(
    score_matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(float(value) for value in row) for row in score_matrix)
    if not matrix or not matrix[0]:
        raise ValueError("score_matrix cannot be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("score_matrix rows must have equal length")
    if any(not isfinite(value) for row in matrix for value in row):
        raise ValueError("score_matrix values must be finite")
    return matrix


def _expected_maximum_normal(trials: int) -> float:
    if trials == 1:
        return 0.0
    probability = (trials - 0.375) / (trials + 0.25)
    return _inverse_normal_cdf(probability)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _inverse_normal_cdf(probability: float) -> float:
    """Acklam inverse-normal approximation for probabilities in ``(0, 1)``."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between zero and one")
    low = 0.02425
    high = 1.0 - low
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.357751867269,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    )
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        0.007784695709041462,
        0.3224671290700398,
        2.445134137142996,
        3.754408661907416,
    )
    if probability < low:
        q = sqrt(-2.0 * log(probability))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return numerator / denominator
    if probability <= high:
        q = probability - 0.5
        r = q * q
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        denominator = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return numerator / denominator
    q = sqrt(-2.0 * log(1.0 - probability))
    numerator = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return numerator / denominator
