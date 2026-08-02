"""Batch-level market-data quality gates.

Cleaning and quality gating intentionally remain separate concerns.  The
cleaner normalizes records and produces an audit report; this module decides
whether that report is acceptable for durable storage and downstream research.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fxbot.data.adapters.base import AdapterDiagnostics
from fxbot.data.cleaning import CleaningReport


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _probability(value: float, field_name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return number


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Acceptance limits for one cleaned historical-data batch.

    Ratios are expressed as fractions in the inclusive range ``0..1``.  A
    ``None`` threshold disables that individual rule.  Fatal issue severities
    and codes always reject the batch regardless of ratio limits.
    """

    require_non_empty: bool = True
    min_output_records: int = 1
    max_rejection_rate: float | None = 0.01
    max_duplicate_rate: float | None = 0.01
    max_spread_rejection_rate: float | None = 0.005
    max_outlier_rate: float | None = 0.01
    max_gap_count: int | None = None
    reject_reordered: bool = False
    require_consistent_diagnostics: bool = True
    reject_adapter_errors: bool = True
    fatal_issue_severities: frozenset[str] = field(
        default_factory=lambda: frozenset({"error", "critical"})
    )
    fatal_issue_codes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.min_output_records < 0:
            raise ValueError("min_output_records must be non-negative")
        if self.max_gap_count is not None and self.max_gap_count < 0:
            raise ValueError("max_gap_count must be non-negative")
        for name in (
            "max_rejection_rate",
            "max_duplicate_rate",
            "max_spread_rejection_rate",
            "max_outlier_rate",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _probability(value, name))
        severities = frozenset(item.strip().lower() for item in self.fatal_issue_severities)
        codes = frozenset(item.strip() for item in self.fatal_issue_codes)
        if any(not item for item in severities):
            raise ValueError("fatal_issue_severities cannot contain empty values")
        if any(not item for item in codes):
            raise ValueError("fatal_issue_codes cannot contain empty values")
        object.__setattr__(self, "fatal_issue_severities", severities)
        object.__setattr__(self, "fatal_issue_codes", codes)


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Normalized metrics used to make an acceptance decision."""

    source_rows_read: int
    source_records_emitted: int
    source_records_rejected: int
    cleaning_input_records: int
    output_records: int
    cleaning_rejections: int
    total_rejections: int
    rejection_rate: float
    duplicate_rate: float
    spread_rejection_rate: float
    outlier_rate: float
    gap_count: int
    reordered: bool


@dataclass(frozen=True, slots=True)
class QualityDecision:
    """Result of evaluating one cleaning report against configured limits."""

    accepted: bool
    reasons: tuple[str, ...]
    metrics: QualityMetrics

    def require_accepted(self) -> None:
        """Raise :class:`QualityGateRejectedError` when the batch was rejected."""

        if not self.accepted:
            raise QualityGateRejectedError(self)


class QualityGateRejectedError(RuntimeError):
    """Raised when a batch fails one or more configured quality rules."""

    def __init__(self, decision: QualityDecision) -> None:
        self.decision = decision
        detail = "; ".join(decision.reasons) or "unspecified quality failure"
        super().__init__(f"Market-data batch rejected: {detail}")


class DataQualityGate:
    """Evaluate cleaning and source diagnostics before durable persistence."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(
        self,
        report: CleaningReport,
        diagnostics: AdapterDiagnostics | None = None,
    ) -> QualityDecision:
        """Return a deterministic acceptance decision for one batch."""

        self._validate_report(report)
        source = diagnostics or AdapterDiagnostics()
        self._validate_diagnostics(source)
        diagnostics_empty = (
            source.rows_read == 0
            and source.records_emitted == 0
            and source.records_rejected == 0
            and not source.errors
        )
        source_rows_read = report.input_records if diagnostics_empty else source.rows_read
        source_records_emitted = (
            report.input_records if diagnostics_empty else source.records_emitted
        )
        cleaning_rejections = report.input_records - report.output_records
        total_rejections = source.records_rejected + cleaning_rejections
        rejection_denominator = max(source_rows_read, report.input_records)

        metrics = QualityMetrics(
            source_rows_read=source_rows_read,
            source_records_emitted=source_records_emitted,
            source_records_rejected=source.records_rejected,
            cleaning_input_records=report.input_records,
            output_records=report.output_records,
            cleaning_rejections=cleaning_rejections,
            total_rejections=total_rejections,
            rejection_rate=_ratio(total_rejections, rejection_denominator),
            duplicate_rate=_ratio(report.duplicates_removed, report.input_records),
            spread_rejection_rate=_ratio(report.spread_rejections, report.input_records),
            outlier_rate=_ratio(report.outliers_detected, report.input_records),
            gap_count=report.gap_count,
            reordered=report.reordered,
        )

        reasons: list[str] = []
        limits = self.thresholds
        if limits.require_non_empty and report.input_records == 0:
            reasons.append("batch contains no input records")
        if report.output_records < limits.min_output_records:
            reasons.append(
                f"output record count {report.output_records} is below "
                f"minimum {limits.min_output_records}"
            )
        self._check_rate(
            reasons,
            "rejection rate",
            metrics.rejection_rate,
            limits.max_rejection_rate,
        )
        self._check_rate(
            reasons,
            "duplicate rate",
            metrics.duplicate_rate,
            limits.max_duplicate_rate,
        )
        self._check_rate(
            reasons,
            "spread rejection rate",
            metrics.spread_rejection_rate,
            limits.max_spread_rejection_rate,
        )
        self._check_rate(
            reasons,
            "outlier rate",
            metrics.outlier_rate,
            limits.max_outlier_rate,
        )
        if limits.max_gap_count is not None and report.gap_count > limits.max_gap_count:
            reasons.append(
                f"gap count {report.gap_count} exceeds maximum {limits.max_gap_count}"
            )
        if limits.reject_reordered and report.reordered:
            reasons.append("records arrived out of chronological order")
        if (
            limits.require_consistent_diagnostics
            and not diagnostics_empty
            and source.records_emitted != report.input_records
        ):
            reasons.append(
                "adapter diagnostics emitted count does not match cleaner input count "
                f"({source.records_emitted} != {report.input_records})"
            )
        if limits.reject_adapter_errors and source.errors:
            reasons.append(f"adapter reported {len(source.errors)} source errors")

        for issue in report.issues:
            if issue.severity.strip().lower() in limits.fatal_issue_severities:
                reasons.append(
                    f"fatal quality issue severity {issue.severity!r}: {issue.code}"
                )
            if issue.code in limits.fatal_issue_codes:
                reasons.append(f"fatal quality issue code: {issue.code}")

        return QualityDecision(
            accepted=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            metrics=metrics,
        )

    @staticmethod
    def _check_rate(
        reasons: list[str],
        label: str,
        actual: float,
        maximum: float | None,
    ) -> None:
        if maximum is not None and actual > maximum:
            reasons.append(f"{label} {actual:.6f} exceeds maximum {maximum:.6f}")

    @staticmethod
    def _validate_report(report: CleaningReport) -> None:
        integer_fields = (
            "input_records",
            "output_records",
            "duplicates_removed",
            "spread_rejections",
            "outliers_detected",
            "outliers_removed",
            "gap_count",
        )
        if any(getattr(report, name) < 0 for name in integer_fields):
            raise ValueError("CleaningReport counters must be non-negative")
        if report.output_records > report.input_records:
            raise ValueError("output_records cannot exceed input_records")
        if report.outliers_removed > report.outliers_detected:
            raise ValueError("outliers_removed cannot exceed outliers_detected")

    @staticmethod
    def _validate_diagnostics(diagnostics: AdapterDiagnostics) -> None:
        if min(
            diagnostics.rows_read,
            diagnostics.records_emitted,
            diagnostics.records_rejected,
        ) < 0:
            raise ValueError("AdapterDiagnostics counters must be non-negative")
        if diagnostics.rows_read > 0 and (
            diagnostics.records_emitted + diagnostics.records_rejected
            > diagnostics.rows_read
        ):
            raise ValueError(
                "AdapterDiagnostics emitted and rejected counts cannot exceed rows_read"
            )
