"""Seeded Monte Carlo trade-path simulation and risk-of-ruin estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from random import Random

from fxbot.backtest.broker import ClosedTrade


class MonteCarloMethod(StrEnum):
    """Supported trade-path resampling methods."""

    RESHUFFLE = "reshuffle"
    BOOTSTRAP = "bootstrap"


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Reproducible simulation and confidence-interval settings."""

    iterations: int = 1_000
    seed: int = 7
    initial_equity: float = 10_000.0
    ruin_threshold_fraction: float = 0.5
    confidence_levels: tuple[float, ...] = (0.05, 0.50, 0.95)

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isfinite(self.initial_equity) or self.initial_equity <= 0.0:
            raise ValueError("initial_equity must be positive and finite")
        if not 0.0 < self.ruin_threshold_fraction < 1.0:
            raise ValueError("ruin_threshold_fraction must be between 0 and 1")
        levels = tuple(float(item) for item in self.confidence_levels)
        if not levels or any(not 0.0 <= item <= 1.0 for item in levels):
            raise ValueError("confidence_levels must be probabilities between 0 and 1")
        if tuple(sorted(set(levels))) != levels:
            raise ValueError("confidence_levels must be sorted and unique")
        object.__setattr__(self, "confidence_levels", levels)


@dataclass(frozen=True, slots=True)
class MonteCarloPath:
    """Summary of one simulated trade ordering or bootstrap sample."""

    final_equity: float
    total_return: float
    maximum_drawdown_amount: float
    maximum_drawdown_fraction: float
    ruined: bool


@dataclass(frozen=True, slots=True)
class MonteCarloQuantile:
    """One percentile across the generated path distribution."""

    probability: float
    final_equity: float
    total_return: float
    maximum_drawdown_amount: float
    maximum_drawdown_fraction: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Complete reproducible Monte Carlo distribution summary."""

    method: MonteCarloMethod
    config: MonteCarloConfig
    paths: tuple[MonteCarloPath, ...]
    quantiles: tuple[MonteCarloQuantile, ...]
    risk_of_ruin: float


def simulate_monte_carlo(
    trades: tuple[ClosedTrade, ...],
    *,
    config: MonteCarloConfig,
    method: MonteCarloMethod = MonteCarloMethod.RESHUFFLE,
) -> MonteCarloResult:
    """Simulate trade-order uncertainty or bootstrap outcome uncertainty."""

    normalized_method = MonteCarloMethod(method)
    pnl = tuple(float(item.net_pnl) for item in trades)
    rng = Random(config.seed)
    paths: list[MonteCarloPath] = []
    for _ in range(config.iterations):
        if normalized_method is MonteCarloMethod.RESHUFFLE:
            sample = list(pnl)
            rng.shuffle(sample)
        else:
            sample = [rng.choice(pnl) for _ in pnl] if pnl else []
        paths.append(_path(tuple(sample), config))

    path_tuple = tuple(paths)
    quantiles = tuple(
        MonteCarloQuantile(
            probability=level,
            final_equity=_quantile(tuple(item.final_equity for item in path_tuple), level),
            total_return=_quantile(tuple(item.total_return for item in path_tuple), level),
            maximum_drawdown_amount=_quantile(
                tuple(item.maximum_drawdown_amount for item in path_tuple),
                level,
            ),
            maximum_drawdown_fraction=_quantile(
                tuple(item.maximum_drawdown_fraction for item in path_tuple),
                level,
            ),
        )
        for level in config.confidence_levels
    )
    ruined = sum(item.ruined for item in path_tuple)
    return MonteCarloResult(
        method=normalized_method,
        config=config,
        paths=path_tuple,
        quantiles=quantiles,
        risk_of_ruin=ruined / len(path_tuple),
    )


def _path(pnl: tuple[float, ...], config: MonteCarloConfig) -> MonteCarloPath:
    equity = config.initial_equity
    peak = equity
    maximum_drawdown = 0.0
    maximum_fraction = 0.0
    threshold = config.initial_equity * config.ruin_threshold_fraction
    ruined = equity <= threshold
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = max(peak - equity, 0.0)
        fraction = drawdown / peak if peak > 0.0 else 1.0
        maximum_drawdown = max(maximum_drawdown, drawdown)
        maximum_fraction = max(maximum_fraction, fraction)
        ruined = ruined or equity <= threshold
    return MonteCarloPath(
        final_equity=equity,
        total_return=equity / config.initial_equity - 1.0,
        maximum_drawdown_amount=maximum_drawdown,
        maximum_drawdown_fraction=maximum_fraction,
        ruined=ruined,
    )


def _quantile(values: tuple[float, ...], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
