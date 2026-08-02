"""Deterministic portfolio risk approval and automatic kill-switch evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from fxbot.risk.limits import PortfolioRiskLimits, RiskLimitCode, RiskViolation
from fxbot.risk.portfolio import (
    PortfolioAnalyzer,
    PortfolioMetrics,
    PortfolioSnapshot,
    TradeProposal,
)


class RiskDecisionStatus(StrEnum):
    """Whether a proposed order may proceed to execution."""

    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Auditable portfolio-level decision for one trade proposal."""

    decision_id: str
    status: RiskDecisionStatus
    proposal_id: str
    evaluated_at: datetime
    metrics_before: PortfolioMetrics
    metrics_after: PortfolioMetrics
    violations: tuple[RiskViolation, ...]
    kill_switch_engaged: bool
    automatic_kill_switch_triggered: bool
    audit_fields: tuple[tuple[str, str], ...]

    @property
    def approved(self) -> bool:
        return self.status is RiskDecisionStatus.APPROVED

    @property
    def rejection_codes(self) -> tuple[RiskLimitCode, ...]:
        return tuple(item.code for item in self.violations)


class RiskGuard:
    """Apply account health, portfolio exposure, and concentration controls."""

    _AUTOMATIC_KILL_CODES = frozenset(
        {
            RiskLimitCode.EQUITY_STOP,
            RiskLimitCode.DAILY_REALIZED_LOSS,
            RiskLimitCode.DAILY_TOTAL_LOSS,
            RiskLimitCode.INTRADAY_DRAWDOWN,
        }
    )

    def __init__(
        self,
        *,
        analyzer: PortfolioAnalyzer,
        limits: PortfolioRiskLimits | None = None,
        policy_name: str = "portfolio-risk",
        policy_version: str = "1",
    ) -> None:
        self.analyzer = analyzer
        self.limits = limits or PortfolioRiskLimits()
        self.policy_name = policy_name.strip() or "portfolio-risk"
        self.policy_version = policy_version.strip() or "1"

    def evaluate(
        self,
        snapshot: PortfolioSnapshot,
        proposal: TradeProposal,
        *,
        evaluated_at: datetime | None = None,
    ) -> RiskDecision:
        """Return a pure, deterministic decision without mutating portfolio state."""

        timestamp = _utc(evaluated_at or snapshot.as_of)
        before = self.analyzer.analyze(
            snapshot,
            include_pending_orders=self.limits.include_pending_order_exposure,
        )
        additional_positions = () if proposal.pending else (proposal.as_position(),)
        additional_orders = (proposal.as_pending_order(),) if proposal.pending else ()
        after = self.analyzer.analyze(
            snapshot,
            additional_positions=additional_positions,
            additional_pending_orders=additional_orders,
            include_pending_orders=self.limits.include_pending_order_exposure,
        )

        violations = [*self._account_violations(snapshot, before, timestamp)]
        violations.extend(self._portfolio_violations(proposal, after))
        violations = _deduplicate_violations(violations)

        automatic_triggered = any(
            item.code in self._AUTOMATIC_KILL_CODES for item in violations
        )
        kill_switch = (
            snapshot.manual_kill_switch
            or snapshot.automatic_kill_switch
            or automatic_triggered
        )
        status = (
            RiskDecisionStatus.APPROVED
            if not violations
            else RiskDecisionStatus.REJECTED
        )
        audit_fields = (
            ("policy_name", self.policy_name),
            ("policy_version", self.policy_version),
            ("snapshot_as_of", snapshot.as_of.isoformat()),
            ("proposal_kind", "pending" if proposal.pending else "market"),
            ("include_pending_exposure", str(self.limits.include_pending_order_exposure)),
        )
        decision_id = self._decision_id(
            snapshot,
            proposal,
            timestamp,
            violations,
        )
        return RiskDecision(
            decision_id=decision_id,
            status=status,
            proposal_id=proposal.proposal_id,
            evaluated_at=timestamp,
            metrics_before=before,
            metrics_after=after,
            violations=tuple(violations),
            kill_switch_engaged=kill_switch,
            automatic_kill_switch_triggered=automatic_triggered,
            audit_fields=audit_fields,
        )

    def _account_violations(
        self,
        snapshot: PortfolioSnapshot,
        metrics: PortfolioMetrics,
        evaluated_at: datetime,
    ) -> list[RiskViolation]:
        limits = self.limits
        violations: list[RiskViolation] = []
        if snapshot.manual_kill_switch:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MANUAL_KILL_SWITCH,
                    message="Manual kill switch is active",
                    observed=snapshot.kill_switch_reason,
                )
            )
        if snapshot.automatic_kill_switch:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.AUTOMATIC_KILL_SWITCH,
                    message="Automatic kill switch is already active",
                    observed=snapshot.kill_switch_reason,
                )
            )

        realized_fraction = (
            metrics.daily_realized_loss_amount / snapshot.day_start_equity
        )
        if realized_fraction >= limits.max_daily_realized_loss_fraction:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.DAILY_REALIZED_LOSS,
                    message="Daily realized-loss limit reached",
                    observed=realized_fraction,
                    limit=limits.max_daily_realized_loss_fraction,
                )
            )
        if metrics.daily_total_loss_fraction >= limits.max_daily_total_loss_fraction:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.DAILY_TOTAL_LOSS,
                    message="Daily total-loss limit reached",
                    observed=metrics.daily_total_loss_fraction,
                    limit=limits.max_daily_total_loss_fraction,
                )
            )
        if metrics.intraday_drawdown_fraction >= limits.max_intraday_drawdown_fraction:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.INTRADAY_DRAWDOWN,
                    message="Intraday peak-to-equity drawdown limit reached",
                    observed=metrics.intraday_drawdown_fraction,
                    limit=limits.max_intraday_drawdown_fraction,
                )
            )
        if metrics.equity_fraction_of_balance <= limits.min_equity_fraction_of_balance:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.EQUITY_STOP,
                    message="Account equity stop reached",
                    observed=metrics.equity_fraction_of_balance,
                    limit=limits.min_equity_fraction_of_balance,
                )
            )

        if snapshot.consecutive_losses >= limits.max_consecutive_losses:
            if snapshot.last_loss_time is None:
                cooldown_active = True
                cooldown_end: str | None = None
            else:
                end = snapshot.last_loss_time + limits.consecutive_loss_cooldown
                cooldown_active = evaluated_at < end
                cooldown_end = end.isoformat()
            if cooldown_active:
                violations.append(
                    RiskViolation(
                        code=RiskLimitCode.CONSECUTIVE_LOSS_COOLDOWN,
                        message="Consecutive-loss cooldown is active",
                        observed=snapshot.consecutive_losses,
                        limit=limits.max_consecutive_losses,
                        scope=cooldown_end,
                    )
                )
        return violations

    def _portfolio_violations(
        self,
        proposal: TradeProposal,
        metrics: PortfolioMetrics,
    ) -> list[RiskViolation]:
        limits = self.limits
        violations: list[RiskViolation] = []

        if metrics.open_position_count > limits.max_open_positions:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_OPEN_POSITIONS,
                    message="Maximum open-position count exceeded",
                    observed=metrics.open_position_count,
                    limit=limits.max_open_positions,
                )
            )
        if metrics.pending_order_count > limits.max_pending_orders:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_PENDING_ORDERS,
                    message="Maximum pending-order count exceeded",
                    observed=metrics.pending_order_count,
                    limit=limits.max_pending_orders,
                )
            )
        symbol_count = metrics.symbol_count(proposal.symbol)
        if symbol_count > limits.max_positions_per_symbol:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_POSITIONS_PER_SYMBOL,
                    message="Maximum exposure count for symbol exceeded",
                    observed=symbol_count,
                    limit=limits.max_positions_per_symbol,
                    scope=proposal.symbol,
                )
            )

        account_equity = metrics.account_equity
        total_risk_fraction = metrics.total_risk_amount / account_equity
        if total_risk_fraction > limits.max_total_risk_fraction:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_TOTAL_RISK,
                    message="Maximum aggregate stop-loss risk exceeded",
                    observed=total_risk_fraction,
                    limit=limits.max_total_risk_fraction,
                )
            )
        gross_multiple = metrics.gross_notional_amount / account_equity
        if gross_multiple > limits.max_gross_notional_multiple:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_GROSS_NOTIONAL,
                    message="Maximum gross notional exposure exceeded",
                    observed=gross_multiple,
                    limit=limits.max_gross_notional_multiple,
                )
            )
        net_multiple = abs(metrics.net_notional_amount) / account_equity
        if net_multiple > limits.max_net_notional_multiple:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_NET_NOTIONAL,
                    message="Maximum absolute net notional exposure exceeded",
                    observed=net_multiple,
                    limit=limits.max_net_notional_multiple,
                )
            )
        if metrics.margin_utilization > limits.max_margin_utilization:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_MARGIN_UTILIZATION,
                    message="Maximum margin utilization exceeded",
                    observed=metrics.margin_utilization,
                    limit=limits.max_margin_utilization,
                )
            )
        concentration = metrics.largest_currency_exposure / account_equity
        if concentration > limits.max_currency_concentration_multiple:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.MAX_CURRENCY_CONCENTRATION,
                    message="Maximum single-currency concentration exceeded",
                    observed=concentration,
                    limit=limits.max_currency_concentration_multiple,
                )
            )
        if limits.require_protective_stop and metrics.unprotected_exposure_count > 0:
            violations.append(
                RiskViolation(
                    code=RiskLimitCode.PROTECTIVE_STOP_REQUIRED,
                    message="Every open or contingent exposure requires a protective stop",
                    observed=metrics.unprotected_exposure_count,
                    limit=0,
                )
            )
        return violations

    def _decision_id(
        self,
        snapshot: PortfolioSnapshot,
        proposal: TradeProposal,
        evaluated_at: datetime,
        violations: list[RiskViolation],
    ) -> str:
        payload = {
            "policy": [self.policy_name, self.policy_version],
            "snapshot": snapshot.as_of.isoformat(),
            "evaluated_at": evaluated_at.isoformat(),
            "proposal": {
                "id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "side": proposal.side.value,
                "volume": proposal.volume,
                "entry": proposal.entry_price,
                "stop": proposal.stop_price,
                "margin": proposal.margin_required,
                "pending": proposal.pending,
            },
            "violations": [item.code.value for item in violations],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("evaluated_at must be timezone-aware")
    return value.astimezone(UTC)


def _deduplicate_violations(items: list[RiskViolation]) -> list[RiskViolation]:
    seen: set[tuple[RiskLimitCode, str | None]] = set()
    result: list[RiskViolation] = []
    for item in items:
        key = (item.code, item.scope)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
