"""PyArrow-backed historical Parquet adapter.

The dependency is imported lazily so the package can still expose CSV and live
adapters in constrained environments. ``pyarrow`` remains a runtime dependency
in the production requirements.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fxbot.data.adapters.base import (
    AdapterDiagnostics,
    HistoricalMarketDataAdapter,
    MarketDataAdapterError,
)
from fxbot.data.parsing import MarketDataRowParser, RowParseError
from fxbot.data.schemas import BarCSVSchema, TickCSVSchema
from fxbot.domain.enums import DataKind, ParseErrorPolicy
from fxbot.domain.models import Bar, HistoricalDataRequest, SymbolSpec, Tick


@dataclass(slots=True)
class _MutableDiagnostics:
    rows_read: int = 0
    records_emitted: int = 0
    records_rejected: int = 0
    errors: list[str] = field(default_factory=list)

    def snapshot(self) -> AdapterDiagnostics:
        return AdapterDiagnostics(
            rows_read=self.rows_read,
            records_emitted=self.records_emitted,
            records_rejected=self.records_rejected,
            errors=tuple(self.errors),
        )


class ParquetMarketDataAdapter(HistoricalMarketDataAdapter):
    """Stream Parquet record batches through the same canonical row parser."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        tick_schema: TickCSVSchema | None = None,
        bar_schema: BarCSVSchema | None = None,
        symbol_specs: Mapping[str, SymbolSpec] | None = None,
        batch_size: int = 65_536,
        error_policy: ParseErrorPolicy = ParseErrorPolicy.RAISE,
        max_diagnostic_errors: int = 100,
    ) -> None:
        self._paths = [str(paths)] if isinstance(paths, (str, Path)) else [str(p) for p in paths]
        self._tick_schema = tick_schema
        self._bar_schema = bar_schema
        self._parser = MarketDataRowParser(symbol_specs)
        self._batch_size = batch_size
        self._error_policy = ParseErrorPolicy(error_policy)
        self._max_diagnostic_errors = max_diagnostic_errors
        self._diagnostics = _MutableDiagnostics()
        if not self._paths:
            raise ValueError("At least one Parquet path is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics.snapshot()

    def iter_ticks(self, request: HistoricalDataRequest) -> Iterator[Tick]:
        if request.kind is not DataKind.TICK:
            raise ValueError("iter_ticks requires a tick HistoricalDataRequest")
        if self._tick_schema is None:
            raise MarketDataAdapterError("tick_schema was not configured")
        self._diagnostics = _MutableDiagnostics()
        for row_number, row in enumerate(self._iter_rows(), start=1):
            try:
                tick = self._parser.parse_tick(row, self._tick_schema, source="parquet")
                if tick.symbol != request.symbol or not request.contains(tick.event_time):
                    continue
                self._diagnostics.records_emitted += 1
                yield tick
            except (RowParseError, ValueError) as exc:
                self._handle_row_error(row_number, exc)

    def iter_bars(self, request: HistoricalDataRequest) -> Iterator[Bar]:
        if request.kind is not DataKind.BAR:
            raise ValueError("iter_bars requires a bar HistoricalDataRequest")
        if self._bar_schema is None:
            raise MarketDataAdapterError("bar_schema was not configured")
        self._diagnostics = _MutableDiagnostics()
        for row_number, row in enumerate(self._iter_rows(), start=1):
            try:
                bar = self._parser.parse_bar(row, self._bar_schema, source="parquet")
                if bar.symbol != request.symbol or bar.timeframe != request.timeframe:
                    continue
                if not request.contains(bar.open_time):
                    continue
                self._diagnostics.records_emitted += 1
                yield bar
            except (RowParseError, ValueError) as exc:
                self._handle_row_error(row_number, exc)

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MarketDataAdapterError(
                "Parquet support requires pyarrow. Install project requirements."
            ) from exc

        try:
            dataset = ds.dataset(self._paths, format="parquet")
            scanner = dataset.scanner(batch_size=self._batch_size)
            for batch in scanner.to_batches():
                for row in batch.to_pylist():
                    self._diagnostics.rows_read += 1
                    yield row
        except Exception as exc:
            raise MarketDataAdapterError(f"Cannot scan Parquet data: {exc}") from exc

    def _handle_row_error(self, row_number: int, exc: Exception) -> None:
        message = f"Parquet row {row_number}: {exc}"
        self._diagnostics.records_rejected += 1
        if len(self._diagnostics.errors) < self._max_diagnostic_errors:
            self._diagnostics.errors.append(message)
        if self._error_policy is ParseErrorPolicy.RAISE:
            raise MarketDataAdapterError(message) from exc
