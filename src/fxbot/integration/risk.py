"""Paper pre-trade authorizer combining quote, session, loss, and exposure gates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from fxbot.execution.models import OrderIntent, RiskDecision
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.production.protections import LossGuard, MarketHoursGuard, QuoteGuard


@dataclass(frozen=True, slots=True)
class PaperExposureLimits:
    max_abs_position_per_symbol: float = 1.0
    max_gross_quantity: float = 3.0

    def __post_init__(self) -> None:
        for name in ("max_abs_position_per_symbol", "max_gross_quantity"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


class GuardedPaperRiskAuthorizer:
    """Apply production protections before every paper order submission."""

    def __init__(
        self,
        *,
        ledger: PaperPortfolioLedger,
        quote_guard: QuoteGuard,
        loss_guard: LossGuard,
        market_hours_guard: MarketHoursGuard | None = None,
        limits: PaperExposureLimits | None = None,
    ) -> None:
        self.ledger = ledger
        self.quote_guard = quote_guard
        self.loss_guard = loss_guard
        self.market_hours_guard = market_hours_guard
        self.limits = limits or PaperExposureLimits()

    def authorize(self, intent: OrderIntent) -> RiskDecision:
        quote = self.ledger.quote(intent.symbol)
        if quote is None:
            return RiskDecision(False, "no executable quote available")

        quote_decision = self.quote_guard.evaluate(quote, now=intent.created_at)
        if not quote_decision.allowed:
            return RiskDecision(False, quote_decision.reason)

        loss_decision = self.loss_guard.evaluate(
            self.ledger.account_risk_snapshot(checked_at=intent.created_at)
        )
        if not loss_decision.allowed and not intent.reduce_only:
            return RiskDecision(False, loss_decision.reason)

        if self.market_hours_guard is not None:
            market_decision = self.market_hours_guard.evaluate(intent.created_at)
            if not market_decision.allowed and not intent.reduce_only:
                return RiskDecision(False, market_decision.reason)

        current = self.ledger.signed_position(intent.symbol)
        if intent.reduce_only:
            if abs(current) <= 1e-12 or current * intent.side.sign >= 0.0:
                return RiskDecision(False, "reduce-only order does not reduce exposure")
            approved = min(intent.quantity, abs(current))
            return RiskDecision(True, "reduce-only exposure reduction approved", approved)

        projected = current + intent.quantity * intent.side.sign
        symbol_room = self.limits.max_abs_position_per_symbol - abs(current)
        gross_room = self.limits.max_gross_quantity - self.ledger.gross_quantity
        approved = min(intent.quantity, max(symbol_room, 0.0), max(gross_room, 0.0))
        if abs(projected) <= self.limits.max_abs_position_per_symbol + 1e-12:
            approved = min(intent.quantity, max(gross_room, 0.0))
        if approved <= 1e-12:
            return RiskDecision(False, "paper exposure limit reached")
        reason = (
            "paper risk approved"
            if approved >= intent.quantity - 1e-12
            else "paper risk reduced quantity to exposure capacity"
        )
        return RiskDecision(True, reason, approved)
