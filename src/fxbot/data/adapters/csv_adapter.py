"""Streaming historical CSV adapter."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from glob import glob
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


class CSVMarketDataAdapter(HistoricalMarketDataAdapter):
    """Read one or more delimited files without loading them fully into memory."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        tick_schema: TickCSVSchema | None = None,
        bar_schema: BarCSVSchema | None = None,
        symbol_specs: Mapping[str, SymbolSpec] | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8-sig",
        error_policy: ParseErrorPolicy = ParseErrorPolicy.RAISE,
        max_diagnostic_errors: int = 100,
    ) -> None:
        self._paths = self._resolve_paths(paths)
        self._tick_schema = tick_schema
        self._bar_schema = bar_schema
        self._parser = MarketDataRowParser(symbol_specs)
        self._delimiter = delimiter
        self._encoding = encoding
        self._error_policy = ParseErrorPolicy(error_policy)
        self._max_diagnostic_errors = max_diagnostic_errors
        self._diagnostics = _MutableDiagnostics()

        if not self._paths:
            raise MarketDataAdapterError(f"No CSV files matched {paths!r}")
        if max_diagnostic_errors < 0:
            raise ValueError("max_diagnostic_errors must be non-negative")

    @property
    def diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics.snapshot()

    def iter_ticks(self, request: HistoricalDataRequest) -> Iterator[Tick]:
        if request.kind is not DataKind.TICK:
            raise ValueError("iter_ticks requires a tick HistoricalDataRequest")
        if self._tick_schema is None:
            raise MarketDataAdapterError("tick_schema was not configured")
        self._diagnostics = _MutableDiagnostics()

        for path, row_number, row in self._iter_rows():
            try:
                tick = self._parser.parse_tick(row, self._tick_schema, source=str(path))
                if tick.symbol != request.symbol or not request.contains(tick.event_time):
                    continue
                self._diagnostics.records_emitted += 1
                yield tick
            except (RowParseError, ValueError) as exc:
                self._handle_row_error(path, row_number, exc)

    def iter_bars(self, request: HistoricalDataRequest) -> Iterator[Bar]:
        if request.kind is not DataKind.BAR:
            raise ValueError("iter_bars requires a bar HistoricalDataRequest")
        if self._bar_schema is None:
            raise MarketDataAdapterError("bar_schema was not configured")
        self._diagnostics = _MutableDiagnostics()

        for path, row_number, row in self._iter_rows():
            try:
                bar = self._parser.parse_bar(row, self._bar_schema, source=str(path))
                if bar.symbol != request.symbol or bar.timeframe != request.timeframe:
                    continue
                if not request.contains(bar.open_time):
                    continue
                self._diagnostics.records_emitted += 1
                yield bar
            except (RowParseError, ValueError) as exc:
                self._handle_row_error(path, row_number, exc)

    def _iter_rows(self) -> Iterator[tuple[Path, int, dict[str, Any]]]:
        for path in self._paths:
            try:
                with path.open("r", encoding=self._encoding, newline="") as handle:
                    reader = csv.DictReader(handle, delimiter=self._delimiter)
                    if reader.fieldnames is None:
                        raise MarketDataAdapterError(f"CSV file {path} has no header")
                    for row_number, row in enumerate(reader, start=2):
                        self._diagnostics.rows_read += 1
                        yield path, row_number, row
            except OSError as exc:
                raise MarketDataAdapterError(f"Cannot read CSV file {path}: {exc}") from exc

    def _handle_row_error(self, path: Path, row_number: int, exc: Exception) -> None:
        message = f"{path}:{row_number}: {exc}"
        self._diagnostics.records_rejected += 1
        if len(self._diagnostics.errors) < self._max_diagnostic_errors:
            self._diagnostics.errors.append(message)
        if self._error_policy is ParseErrorPolicy.RAISE:
            raise MarketDataAdapterError(message) from exc

    @staticmethod
    def _resolve_paths(paths: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        candidates = [paths] if isinstance(paths, (str, Path)) else list(paths)
        resolved: set[Path] = set()
        for candidate in candidates:
            text = str(candidate)
            matches = glob(text, recursive=True)
            if matches:
                resolved.update(Path(match).resolve() for match in matches if Path(match).is_file())
            else:
                path = Path(candidate).expanduser().resolve()
                if path.is_file():
                    resolved.add(path)
        return tuple(sorted(resolved))
