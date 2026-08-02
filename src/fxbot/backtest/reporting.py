"""Machine-readable performance reports and strategy/symbol breakdowns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import inf
from typing import Any

from fxbot.backtest.broker import ClosedTrade
from fxbot.backtest.drawdown import DrawdownAnalysis, analyze_drawdowns
from fxbot.backtest.metrics import (
    BenchmarkComparison,
    PerformanceMetrics,
    PeriodReturn,
    calculate_performance,
    calendar_returns,
    compare_benchmark,
)
from fxbot.backtest.monte_carlo import (
    MonteCarloConfig,
    MonteCarloMethod,
    MonteCarloResult,
    simulate_monte_carlo,
)
from fxbot.backtest.results import BacktestResult, EquityPoint
from fxbot.backtest.trades import TradeStatistics, analyze_trades


@dataclass(frozen=True, slots=True)
class PerformanceBreakdown:
    """Trade statistics for one symbol, side, strategy, or other grouping key."""

    key: str
    statistics: TradeStatistics


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Complete deterministic backtest analytics report."""

    generated_at: datetime
    result_fingerprint: str
    metrics: PerformanceMetrics
    drawdowns: DrawdownAnalysis
    monthly_returns: tuple[PeriodReturn, ...]
    yearly_returns: tuple[PeriodReturn, ...]
    symbol_breakdown: tuple[PerformanceBreakdown, ...]
    side_breakdown: tuple[PerformanceBreakdown, ...]
    strategy_breakdown: tuple[PerformanceBreakdown, ...]
    benchmark: BenchmarkComparison | None = None
    monte_carlo: MonteCarloResult | None = None

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        object.__setattr__(self, "generated_at", self.generated_at.astimezone(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with explicit infinity handling."""

        return _json_safe(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report using stable key ordering."""

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_performance_report(
    result: BacktestResult,
    *,
    annual_risk_free_rate: float = 0.0,
    monte_carlo_config: MonteCarloConfig | None = None,
    monte_carlo_method: MonteCarloMethod = MonteCarloMethod.RESHUFFLE,
    strategy_by_trade_id: Mapping[str, str] | None = None,
    benchmark_curve: tuple[EquityPoint, ...] | None = None,
    generated_at: datetime | None = None,
) -> PerformanceReport:
    """Build a complete report from an immutable Phase 4A result."""

    return PerformanceReport(
        generated_at=generated_at or result.ended_at,
        result_fingerprint=result.fingerprint,
        metrics=calculate_performance(
            result,
            annual_risk_free_rate=annual_risk_free_rate,
        ),
        drawdowns=analyze_drawdowns(result.equity_curve),
        monthly_returns=calendar_returns(result.equity_curve, frequency="month"),
        yearly_returns=calendar_returns(result.equity_curve, frequency="year"),
        symbol_breakdown=_breakdown(result.trades, key=lambda item: item.symbol),
        side_breakdown=_breakdown(result.trades, key=lambda item: item.side.value),
        strategy_breakdown=_strategy_breakdown(result.trades, strategy_by_trade_id),
        benchmark=(
            compare_benchmark(result.equity_curve, benchmark_curve)
            if benchmark_curve is not None
            else None
        ),
        monte_carlo=(
            simulate_monte_carlo(
                result.trades,
                config=monte_carlo_config,
                method=monte_carlo_method,
            )
            if monte_carlo_config is not None
            else None
        ),
    )


def _breakdown(
    trades: tuple[ClosedTrade, ...],
    *,
    key: Callable[[ClosedTrade], str],
) -> tuple[PerformanceBreakdown, ...]:
    grouped: dict[str, list[ClosedTrade]] = {}
    for trade in trades:
        group = str(key(trade))
        grouped.setdefault(group, []).append(trade)
    return tuple(
        PerformanceBreakdown(group, analyze_trades(tuple(grouped[group])))
        for group in sorted(grouped)
    )



def _strategy_breakdown(
    trades: tuple[ClosedTrade, ...],
    mapping: Mapping[str, str] | None,
) -> tuple[PerformanceBreakdown, ...]:
    if mapping is None:
        return ()
    known = {item.trade_id for item in trades}
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(f"Strategy mapping references unknown trades: {sorted(unknown)}")
    grouped: dict[str, list[ClosedTrade]] = {}
    for trade in trades:
        strategy = mapping.get(trade.trade_id)
        if strategy is None:
            continue
        normalized = strategy.strip()
        if not normalized:
            raise ValueError("Strategy names cannot be empty")
        grouped.setdefault(normalized, []).append(trade)
    return tuple(
        PerformanceBreakdown(group, analyze_trades(tuple(grouped[group])))
        for group in sorted(grouped)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and value in {inf, -inf}:
        return "inf" if value > 0.0 else "-inf"
    return value
