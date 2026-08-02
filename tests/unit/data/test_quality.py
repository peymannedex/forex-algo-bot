from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxbot.data.adapters.base import AdapterDiagnostics
from fxbot.data.cleaning import CleaningReport, DataQualityIssue
from fxbot.data.quality import (
    DataQualityGate,
    QualityGateRejectedError,
    QualityThresholds,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_quality_gate_accepts_healthy_batch() -> None:
    report = CleaningReport(input_records=100, output_records=100)
    diagnostics = AdapterDiagnostics(rows_read=100, records_emitted=100)

    decision = DataQualityGate().evaluate(report, diagnostics)

    assert decision.accepted
    assert decision.reasons == ()
    assert decision.metrics.rejection_rate == 0.0
    assert decision.metrics.output_records == 100
    decision.require_accepted()


def test_quality_gate_rejects_empty_and_below_minimum() -> None:
    gate = DataQualityGate(QualityThresholds(min_output_records=2))

    decision = gate.evaluate(CleaningReport(input_records=0, output_records=0))

    assert not decision.accepted
    assert "batch contains no input records" in decision.reasons
    assert any("below minimum 2" in reason for reason in decision.reasons)
    with pytest.raises(QualityGateRejectedError, match="rejected"):
        decision.require_accepted()


def test_quality_gate_enforces_rates_gaps_and_ordering() -> None:
    gate = DataQualityGate(
        QualityThresholds(
            max_rejection_rate=0.10,
            max_duplicate_rate=0.05,
            max_spread_rejection_rate=0.01,
            max_outlier_rate=0.01,
            max_gap_count=1,
            reject_reordered=True,
        )
    )
    report = CleaningReport(
        input_records=100,
        output_records=80,
        duplicates_removed=10,
        spread_rejections=5,
        outliers_detected=2,
        reordered=True,
        gap_count=2,
    )

    decision = gate.evaluate(report)

    assert not decision.accepted
    assert decision.metrics.rejection_rate == pytest.approx(0.20)
    assert decision.metrics.duplicate_rate == pytest.approx(0.10)
    assert decision.metrics.spread_rejection_rate == pytest.approx(0.05)
    assert decision.metrics.outlier_rate == pytest.approx(0.02)
    assert len(decision.reasons) == 6


def test_quality_gate_includes_source_rejections() -> None:
    report = CleaningReport(input_records=95, output_records=95)
    diagnostics = AdapterDiagnostics(
        rows_read=100,
        records_emitted=95,
        records_rejected=5,
    )
    gate = DataQualityGate(QualityThresholds(max_rejection_rate=0.04))

    decision = gate.evaluate(report, diagnostics)

    assert decision.metrics.total_rejections == 5
    assert decision.metrics.rejection_rate == pytest.approx(0.05)
    assert not decision.accepted


def test_quality_gate_rejects_inconsistent_adapter_diagnostics() -> None:
    report = CleaningReport(input_records=10, output_records=10)
    diagnostics = AdapterDiagnostics(rows_read=12, records_emitted=11, records_rejected=1)

    decision = DataQualityGate().evaluate(report, diagnostics)

    assert not decision.accepted
    assert any("does not match cleaner input" in reason for reason in decision.reasons)

    permissive = DataQualityGate(
        QualityThresholds(
            require_consistent_diagnostics=False,
            max_rejection_rate=None,
        )
    ).evaluate(report, diagnostics)
    assert permissive.accepted



def test_quality_gate_rejects_adapter_errors_and_zero_emission_mismatch() -> None:
    report = CleaningReport(input_records=2, output_records=2)
    diagnostics = AdapterDiagnostics(
        rows_read=2,
        records_emitted=0,
        errors=("source parse failure",),
    )

    decision = DataQualityGate().evaluate(report, diagnostics)

    assert not decision.accepted
    assert any("does not match cleaner input" in reason for reason in decision.reasons)
    assert "adapter reported 1 source errors" in decision.reasons


def test_quality_gate_rejects_fatal_issue_severity_and_code() -> None:
    report = CleaningReport(
        input_records=2,
        output_records=2,
        issues=(
            DataQualityIssue("clock_error", "clock moved", severity="ERROR"),
            DataQualityIssue("vendor_halt", "source halted", timestamp=NOW),
        ),
    )
    gate = DataQualityGate(
        QualityThresholds(fatal_issue_codes=frozenset({"vendor_halt"}))
    )

    decision = gate.evaluate(report)

    assert not decision.accepted
    assert any("severity" in reason for reason in decision.reasons)
    assert "fatal quality issue code: vendor_halt" in decision.reasons


def test_quality_thresholds_validate_configuration() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        QualityThresholds(max_rejection_rate=1.1)
    with pytest.raises(ValueError, match="min_output_records"):
        QualityThresholds(min_output_records=-1)
    with pytest.raises(ValueError, match="max_gap_count"):
        QualityThresholds(max_gap_count=-1)
    with pytest.raises(ValueError, match="empty"):
        QualityThresholds(fatal_issue_codes=frozenset({""}))


def test_quality_gate_validates_report_and_diagnostics_invariants() -> None:
    gate = DataQualityGate()
    with pytest.raises(ValueError, match="non-negative"):
        gate.evaluate(CleaningReport(input_records=-1, output_records=0))
    with pytest.raises(ValueError, match="cannot exceed"):
        gate.evaluate(CleaningReport(input_records=1, output_records=2))
    with pytest.raises(ValueError, match="outliers_removed"):
        gate.evaluate(
            CleaningReport(
                input_records=2,
                output_records=1,
                outliers_detected=0,
                outliers_removed=1,
            )
        )
    with pytest.raises(ValueError, match="cannot exceed rows_read"):
        gate.evaluate(
            CleaningReport(input_records=1, output_records=1),
            AdapterDiagnostics(rows_read=1, records_emitted=1, records_rejected=1),
        )
