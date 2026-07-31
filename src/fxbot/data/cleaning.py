"""Conservative market-data cleaning and quality reporting.

The cleaner removes only deterministic defects by default: exact duplicates,
out-of-order records, and invalid/abnormal spreads when explicitly configured.
Return outliers are flagged, not dropped, unless the caller opts in. This is
important because central-bank and macro releases can produce genuine jumps.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import log
from statistics import median
from typing import Generic, TypeVar

from fxbot.domain.models import Bar, SymbolSpec, Tick

T = TypeVar("T", Tick, Bar)


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    """A non-fatal quality observation produced during cleaning."""

    code: str
    message: str
    symbol: str | None = None
    timestamp: datetime | None = None
    severity: str = "warning"


@dataclass(frozen=True, slots=True)
class DataCleaningConfig:
    """Cleaning controls. Destructive statistical filters default to disabled."""

    sort_records: bool = True
    remove_exact_duplicates: bool = True
    reject_locked_quotes: bool = False
    max_spread_pips: float | None = None
    max_gap_seconds: float | None = None
    detect_return_outliers: bool = False
    return_mad_z_threshold: float = 12.0
    drop_return_outliers: bool = False

    def __post_init__(self) -> None:
        if self.max_spread_pips is not None and self.max_spread_pips <= 0:
            raise ValueError("max_spread_pips must be positive")
        if self.max_gap_seconds is not None and self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if self.return_mad_z_threshold <= 0:
            raise ValueError("return_mad_z_threshold must be positive")
        if self.drop_return_outliers and not self.detect_return_outliers:
            raise ValueError("drop_return_outliers requires detect_return_outliers=True")


@dataclass(frozen=True, slots=True)
class CleaningReport:
    input_records: int
    output_records: int
    duplicates_removed: int = 0
    spread_rejections: int = 0
    outliers_detected: int = 0
    outliers_removed: int = 0
    reordered: bool = False
    gap_count: int = 0
    issues: tuple[DataQualityIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class CleanedBatch(Generic[T]):
    records: tuple[T, ...]
    report: CleaningReport


@dataclass(slots=True)
class _Counters:
    duplicates: int = 0
    spread_rejections: int = 0
    outliers_detected: int = 0
    outliers_removed: int = 0
    gaps: int = 0
    issues: list[DataQualityIssue] = field(default_factory=list)


class MarketDataCleaner:
    """Clean validated ticks or bars while preserving a complete audit report."""

    def __init__(
        self,
        config: DataCleaningConfig | None = None,
        symbol_specs: Mapping[str, SymbolSpec] | None = None,
    ) -> None:
        self.config = config or DataCleaningConfig()
        self._symbol_specs = {key.upper(): value for key, value in (symbol_specs or {}).items()}

    def clean_ticks(self, records: Iterable[Tick]) -> CleanedBatch[Tick]:
        items = list(records)
        counters = _Counters()
        original_order = [item.event_time for item in items]

        if self.config.sort_records:
            items.sort(
                key=lambda item: (
                    item.symbol,
                    item.event_time,
                    -1 if item.sequence is None else item.sequence,
                )
            )
        reordered = original_order != [item.event_time for item in items]

        if self.config.remove_exact_duplicates:
            items = self._deduplicate(
                items,
                key=lambda item: (
                    item.symbol,
                    item.event_time,
                    item.bid,
                    item.ask,
                    item.sequence,
                ),
                counters=counters,
            )

        items = self._filter_tick_spreads(items, counters)
        self._detect_gaps(items, lambda item: item.event_time, counters)
        if self.config.detect_return_outliers:
            items = self._handle_tick_outliers(items, counters)

        return CleanedBatch(
            records=tuple(items),
            report=CleaningReport(
                input_records=len(original_order),
                output_records=len(items),
                duplicates_removed=counters.duplicates,
                spread_rejections=counters.spread_rejections,
                outliers_detected=counters.outliers_detected,
                outliers_removed=counters.outliers_removed,
                reordered=reordered,
                gap_count=counters.gaps,
                issues=tuple(counters.issues),
            ),
        )

    def clean_bars(self, records: Iterable[Bar]) -> CleanedBatch[Bar]:
        items = list(records)
        counters = _Counters()
        original_order = [item.open_time for item in items]

        if self.config.sort_records:
            items.sort(key=lambda item: (item.open_time, item.symbol, item.timeframe.value))
        reordered = original_order != [item.open_time for item in items]

        if self.config.remove_exact_duplicates:
            items = self._deduplicate(
                items,
                key=lambda item: (
                    item.symbol,
                    item.timeframe,
                    item.open_time,
                    item.bid,
                    item.ask,
                ),
                counters=counters,
            )

        items = self._filter_bar_spreads(items, counters)
        self._detect_bar_gaps(items, counters)

        return CleanedBatch(
            records=tuple(items),
            report=CleaningReport(
                input_records=len(original_order),
                output_records=len(items),
                duplicates_removed=counters.duplicates,
                spread_rejections=counters.spread_rejections,
                reordered=reordered,
                gap_count=counters.gaps,
                issues=tuple(counters.issues),
            ),
        )

    @staticmethod
    def _deduplicate(
        items: list[T],
        *,
        key: Callable[[T], object],
        counters: _Counters,
    ) -> list[T]:
        seen: set[object] = set()
        result: list[T] = []
        for item in items:
            marker = key(item)
            if marker in seen:
                counters.duplicates += 1
                continue
            seen.add(marker)
            result.append(item)
        return result

    def _filter_tick_spreads(self, items: list[Tick], counters: _Counters) -> list[Tick]:
        result: list[Tick] = []
        for item in items:
            if self.config.reject_locked_quotes and item.ask == item.bid:
                counters.spread_rejections += 1
                counters.issues.append(
                    DataQualityIssue(
                        code="locked_quote",
                        message="Locked bid/ask quote rejected",
                        symbol=item.symbol,
                        timestamp=item.event_time,
                    )
                )
                continue
            if self._spread_exceeds_limit(item.symbol, item.spread):
                counters.spread_rejections += 1
                counters.issues.append(
                    DataQualityIssue(
                        code="spread_limit",
                        message="Tick spread exceeded configured maximum",
                        symbol=item.symbol,
                        timestamp=item.event_time,
                    )
                )
                continue
            result.append(item)
        return result

    def _filter_bar_spreads(self, items: list[Bar], counters: _Counters) -> list[Bar]:
        result: list[Bar] = []
        for item in items:
            locked = item.spread_open == 0.0 or item.spread_close == 0.0
            spread = max(item.spread_open, item.spread_close)
            if self.config.reject_locked_quotes and locked:
                counters.spread_rejections += 1
                counters.issues.append(
                    DataQualityIssue(
                        code="locked_bar_quote",
                        message="Bar with locked bid/ask open or close rejected",
                        symbol=item.symbol,
                        timestamp=item.open_time,
                    )
                )
                continue
            if self._spread_exceeds_limit(item.symbol, spread):
                counters.spread_rejections += 1
                counters.issues.append(
                    DataQualityIssue(
                        code="spread_limit",
                        message="Bar spread exceeded configured maximum",
                        symbol=item.symbol,
                        timestamp=item.open_time,
                    )
                )
                continue
            result.append(item)
        return result

    def _spread_exceeds_limit(self, symbol: str, spread: float) -> bool:
        limit = self.config.max_spread_pips
        if limit is None:
            return False
        try:
            spec = self._symbol_specs[symbol.upper()]
        except KeyError as exc:
            raise ValueError(
                f"SymbolSpec for {symbol!r} is required when max_spread_pips is set"
            ) from exc
        return spread / spec.pip_size > limit

    def _detect_gaps(
        self,
        items: list[T],
        timestamp: Callable[[T], datetime],
        counters: _Counters,
    ) -> None:
        threshold = self.config.max_gap_seconds
        if threshold is None:
            return
        previous_by_symbol: dict[str, T] = {}
        for current in items:
            previous = previous_by_symbol.get(current.symbol)
            previous_by_symbol[current.symbol] = current
            if previous is None:
                continue
            delta = (timestamp(current) - timestamp(previous)).total_seconds()
            if delta > threshold:
                counters.gaps += 1
                counters.issues.append(
                    DataQualityIssue(
                        code="time_gap",
                        message=f"Observed data gap of {delta:.3f} seconds",
                        symbol=current.symbol,
                        timestamp=timestamp(current),
                    )
                )

    def _detect_bar_gaps(self, items: list[Bar], counters: _Counters) -> None:
        previous_by_series: dict[tuple[str, object], Bar] = {}
        for current in items:
            key = (current.symbol, current.timeframe)
            previous = previous_by_series.get(key)
            previous_by_series[key] = current
            if previous is None:
                continue
            expected = previous.timeframe.seconds
            if expected is None:
                continue
            actual = (current.open_time - previous.open_time).total_seconds()
            if actual > expected * 1.5:
                counters.gaps += 1
                counters.issues.append(
                    DataQualityIssue(
                        code="missing_bar_interval",
                        message=(
                            f"Expected approximately {expected}s between bars, "
                            f"observed {actual:.0f}s"
                        ),
                        symbol=current.symbol,
                        timestamp=current.open_time,
                    )
                )

    def _handle_tick_outliers(self, items: list[Tick], counters: _Counters) -> list[Tick]:
        flagged_global_indices: set[int] = set()
        indices_by_symbol: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            indices_by_symbol.setdefault(item.symbol, []).append(index)

        for indices in indices_by_symbol.values():
            if len(indices) < 5:
                continue
            series = [items[index] for index in indices]
            returns = [
                log(current.mid / previous.mid)
                for previous, current in zip(series, series[1:], strict=False)
            ]
            center = median(returns)
            absolute_deviations = [abs(value - center) for value in returns]
            mad = median(absolute_deviations)
            if mad == 0.0:
                continue

            # 0.67448975 makes the MAD score comparable to a standard z-score
            # under a normal distribution. The index offset maps a return to
            # the later tick that generated it.
            for local_index, value in enumerate(returns, start=1):
                score = 0.67448975 * abs(value - center) / mad
                if score > self.config.return_mad_z_threshold:
                    flagged_global_indices.add(indices[local_index])

        if not flagged_global_indices:
            return items

        counters.outliers_detected += len(flagged_global_indices)
        for index in sorted(flagged_global_indices):
            item = items[index]
            counters.issues.append(
                DataQualityIssue(
                    code="return_outlier",
                    message="Robust log-return threshold exceeded",
                    symbol=item.symbol,
                    timestamp=item.event_time,
                    severity="info" if not self.config.drop_return_outliers else "warning",
                )
            )

        if not self.config.drop_return_outliers:
            return items
        counters.outliers_removed += len(flagged_global_indices)
        return [
            item
            for index, item in enumerate(items)
            if index not in flagged_global_indices
        ]
