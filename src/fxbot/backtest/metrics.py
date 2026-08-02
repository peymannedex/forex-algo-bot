"""Portfolio performance metrics and calendar return aggregation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import inf, isfinite, sqrt
from statistics import fmean, median, stdev

from fxbot.backtest.drawdown import DrawdownAnalysis, analyze_drawdowns
from fxbot.backtest.results import BacktestResult, EquityPoint
from fxbot.backtest.trades import TradeStatistics, analyze_trades

_SECONDS_PER_YEAR = 365.2425 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True, slots=True)
class PeriodReturn:
    """Compounded return assigned to one calendar period."""

    period: str
    return_fraction: float




@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Relative performance and return-series relationship to a benchmark."""

    strategy_total_return: float
    benchmark_total_return: float
    excess_total_return: float
    annualized_tracking_error: float
    information_ratio: float
    beta: float
    annualized_alpha: float
    correlation: float


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Risk-adjusted and absolute performance statistics for one backtest."""

    initial_equity: float
    final_equity: float
    net_profit: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    maximum_drawdown_amount: float
    maximum_drawdown_fraction: float
    maximum_drawdown_duration_seconds: float
    exposure_fraction: float
    time_in_market_seconds: float
    trade_statistics: TradeStatistics


def calculate_performance(
    result: BacktestResult,
    *,
    annual_risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate deterministic metrics from a complete backtest result."""

    if not isfinite(annual_risk_free_rate):
        raise ValueError("annual_risk_free_rate must be finite")
    returns = equity_returns(result.equity_curve)
    periods_per_year = _infer_periods_per_year(result.equity_curve)
    cagr = _cagr(result)
    volatility = _annualized_volatility(returns, periods_per_year)
    sharpe = _sharpe(returns, periods_per_year, annual_risk_free_rate)
    sortino = _sortino(returns, periods_per_year, annual_risk_free_rate)
    drawdown = analyze_drawdowns(result.equity_curve)
    stats = analyze_trades(result.trades)
    net_profit = result.final_equity - result.initial_equity
    exposure, seconds = _exposure(result.equity_curve)
    return PerformanceMetrics(
        initial_equity=result.initial_equity,
        final_equity=result.final_equity,
        net_profit=net_profit,
        total_return=result.total_return,
        cagr=cagr,
        annualized_volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=_ratio(cagr, drawdown.maximum_fraction),
        recovery_factor=_ratio(net_profit, drawdown.maximum_amount),
        maximum_drawdown_amount=drawdown.maximum_amount,
        maximum_drawdown_fraction=drawdown.maximum_fraction,
        maximum_drawdown_duration_seconds=drawdown.maximum_duration_seconds,
        exposure_fraction=exposure,
        time_in_market_seconds=seconds,
        trade_statistics=stats,
    )


def equity_returns(equity_curve: tuple[EquityPoint, ...]) -> tuple[float, ...]:
    """Return simple point-to-point equity returns, rejecting non-positive bases."""

    output: list[float] = []
    for previous, current in itertools.pairwise(equity_curve):
        if previous.equity <= 0.0:
            raise ValueError("Cannot calculate returns from non-positive equity")
        output.append(current.equity / previous.equity - 1.0)
    return tuple(output)


def calendar_returns(
    equity_curve: tuple[EquityPoint, ...],
    *,
    frequency: str,
) -> tuple[PeriodReturn, ...]:
    """Compound equity returns by calendar month or calendar year."""

    if frequency not in {"month", "year"}:
        raise ValueError("frequency must be 'month' or 'year'")
    grouped: dict[str, float] = {}
    for previous, current in itertools.pairwise(equity_curve):
        if previous.equity <= 0.0:
            raise ValueError("Cannot calculate returns from non-positive equity")
        key = (
            current.timestamp.strftime("%Y-%m")
            if frequency == "month"
            else current.timestamp.strftime("%Y")
        )
        step_return = current.equity / previous.equity - 1.0
        grouped[key] = (1.0 + grouped.get(key, 0.0)) * (1.0 + step_return) - 1.0
    return tuple(PeriodReturn(key, grouped[key]) for key in sorted(grouped))



def compare_benchmark(
    strategy_curve: tuple[EquityPoint, ...],
    benchmark_curve: tuple[EquityPoint, ...],
) -> BenchmarkComparison:
    """Compare synchronized strategy and benchmark equity curves."""

    if len(strategy_curve) != len(benchmark_curve) or len(strategy_curve) < 2:
        raise ValueError("Strategy and benchmark curves must have equal length of at least two")
    if any(
        strategy.timestamp != benchmark.timestamp
        for strategy, benchmark in zip(strategy_curve, benchmark_curve, strict=True)
    ):
        raise ValueError("Strategy and benchmark timestamps must match")
    strategy_returns = equity_returns(strategy_curve)
    benchmark_returns = equity_returns(benchmark_curve)
    periods_per_year = _infer_periods_per_year(strategy_curve)
    excess = tuple(
        strategy - benchmark
        for strategy, benchmark in zip(strategy_returns, benchmark_returns, strict=True)
    )
    tracking_error = (
        stdev(excess) * sqrt(periods_per_year)
        if len(excess) >= 2 and periods_per_year > 0.0
        else 0.0
    )
    covariance = _covariance(strategy_returns, benchmark_returns)
    benchmark_variance = _variance(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance > 0.0 else 0.0
    strategy_mean = fmean(strategy_returns)
    benchmark_mean = fmean(benchmark_returns)
    alpha = (strategy_mean - beta * benchmark_mean) * periods_per_year
    strategy_total = strategy_curve[-1].equity / strategy_curve[0].equity - 1.0
    benchmark_total = benchmark_curve[-1].equity / benchmark_curve[0].equity - 1.0
    return BenchmarkComparison(
        strategy_total_return=strategy_total,
        benchmark_total_return=benchmark_total,
        excess_total_return=strategy_total - benchmark_total,
        annualized_tracking_error=tracking_error,
        information_ratio=_ratio(
            fmean(excess) * periods_per_year,
            tracking_error,
        ),
        beta=beta,
        annualized_alpha=alpha,
        correlation=_correlation(strategy_returns, benchmark_returns),
    )


def drawdown_metrics(result: BacktestResult) -> DrawdownAnalysis:
    """Convenience wrapper for report builders."""

    return analyze_drawdowns(result.equity_curve)


def _cagr(result: BacktestResult) -> float:
    elapsed = (result.ended_at - result.started_at).total_seconds()
    if elapsed <= 0.0 or result.initial_equity <= 0.0 or result.final_equity <= 0.0:
        return 0.0
    years = elapsed / _SECONDS_PER_YEAR
    return (result.final_equity / result.initial_equity) ** (1.0 / years) - 1.0


def _infer_periods_per_year(equity_curve: tuple[EquityPoint, ...]) -> float:
    intervals = tuple(
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in itertools.pairwise(equity_curve)
        if current.timestamp > previous.timestamp
    )
    if not intervals:
        return 0.0
    return _SECONDS_PER_YEAR / float(median(intervals))


def _annualized_volatility(returns: tuple[float, ...], periods_per_year: float) -> float:
    if len(returns) < 2 or periods_per_year <= 0.0:
        return 0.0
    return stdev(returns) * sqrt(periods_per_year)


def _sharpe(
    returns: tuple[float, ...],
    periods_per_year: float,
    annual_risk_free_rate: float,
) -> float:
    if len(returns) < 2 or periods_per_year <= 0.0:
        return 0.0
    risk_free_period = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = tuple(value - risk_free_period for value in returns)
    deviation = stdev(excess)
    return _ratio(fmean(excess) * sqrt(periods_per_year), deviation)


def _sortino(
    returns: tuple[float, ...],
    periods_per_year: float,
    annual_risk_free_rate: float,
) -> float:
    if not returns or periods_per_year <= 0.0:
        return 0.0
    risk_free_period = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = tuple(value - risk_free_period for value in returns)
    downside = sqrt(fmean(min(value, 0.0) ** 2 for value in excess))
    return _ratio(fmean(excess) * sqrt(periods_per_year), downside)


def _exposure(equity_curve: tuple[EquityPoint, ...]) -> tuple[float, float]:
    if len(equity_curve) < 2:
        return 0.0, 0.0
    total = 0.0
    exposed = 0.0
    for current, following in itertools.pairwise(equity_curve):
        duration = (following.timestamp - current.timestamp).total_seconds()
        if duration < 0.0:
            raise ValueError("equity_curve must be chronologically ordered")
        total += duration
        if current.margin_used > 0.0:
            exposed += duration
    return (exposed / total if total > 0.0 else 0.0, exposed)



def _variance(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _covariance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / (len(left) - 1)


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_variance = _variance(left)
    right_variance = _variance(right)
    denominator = sqrt(left_variance * right_variance)
    return _covariance(left, right) / denominator if denominator > 0.0 else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return inf if numerator > 0.0 else 0.0
