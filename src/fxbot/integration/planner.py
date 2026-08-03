"""Convert strategy decisions into deterministic, risk-sized order intents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from fxbot.execution.models import (
    OrderIntent,
    OrderSide,
    OrderType,
    Quote,
    TimeInForce,
)
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.risk.models import (
    InstrumentRiskSpec,
    PositionSizingRequest,
    SizingMethod,
    TradeSide,
)
from fxbot.risk.position_sizing import PositionSizer
from fxbot.strategy.models import SignalAction, StrategyDecision


class QuantityPolicy(Protocol):
    """Resolve executable lot quantity for one directional strategy decision."""

    def quantity(
        self,
        decision: StrategyDecision,
        quote: Quote,
        ledger: PaperPortfolioLedger,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class FixedQuantityPolicy:
    quantity_value: float = 0.01

    def __post_init__(self) -> None:
        if self.quantity_value <= 0.0:
            raise ValueError("quantity_value must be positive")

    def quantity(
        self,
        decision: StrategyDecision,
        quote: Quote,
        ledger: PaperPortfolioLedger,
    ) -> float:
        del decision, quote, ledger
        return float(self.quantity_value)


class PositionSizerQuantityPolicy:
    """Bridge Phase 2 position sizing into strategy-to-execution planning."""

    def __init__(
        self,
        *,
        sizer: PositionSizer,
        instruments: dict[str, InstrumentRiskSpec],
        risk_fraction: float = 0.005,
    ) -> None:
        if not 0.0 < risk_fraction <= 1.0:
            raise ValueError("risk_fraction must be between zero and one")
        self.sizer = sizer
        self.instruments = {
            symbol.strip().upper(): spec for symbol, spec in instruments.items()
        }
        self.risk_fraction = float(risk_fraction)

    def quantity(
        self,
        decision: StrategyDecision,
        quote: Quote,
        ledger: PaperPortfolioLedger,
    ) -> float:
        instrument = self.instruments.get(decision.symbol)
        if instrument is None:
            raise KeyError(f"No risk instrument configured for {decision.symbol}")
        if decision.stop_loss is None:
            return 0.0
        side = (
            TradeSide.LONG
            if decision.action is SignalAction.BUY
            else TradeSide.SHORT
        )
        entry_price = quote.ask if side is TradeSide.LONG else quote.bid
        result = self.sizer.size(
            PositionSizingRequest(
                account=ledger.account_snapshot(),
                instrument=instrument,
                side=side,
                entry_price=entry_price,
                stop_price=decision.stop_loss,
                method=SizingMethod.FIXED_FRACTIONAL,
                risk_fraction=self.risk_fraction,
            )
        )
        return result.normalized_volume if result.accepted else 0.0


class DecisionOrderPlanner:
    """Plan net-position market orders from canonical strategy decisions."""

    def __init__(
        self,
        quantity_policy: QuantityPolicy,
        *,
        allow_pyramiding: bool = False,
    ) -> None:
        self.quantity_policy = quantity_policy
        self.allow_pyramiding = allow_pyramiding

    def plan(
        self,
        decision: StrategyDecision,
        quote: Quote,
        ledger: PaperPortfolioLedger,
    ) -> tuple[OrderIntent, ...]:
        if decision.symbol != quote.symbol:
            raise ValueError("decision and quote symbols must match")
        if decision.action is SignalAction.HOLD:
            return ()

        current = ledger.signed_position(decision.symbol)
        if decision.action is SignalAction.EXIT:
            return self._exit_intents(decision, current)

        side = (
            OrderSide.BUY
            if decision.action is SignalAction.BUY
            else OrderSide.SELL
        )
        target_sign = side.sign
        intents: list[OrderIntent] = []
        if current * target_sign < 0.0:
            intents.extend(self._exit_intents(decision, current))
            current = 0.0
        if current * target_sign > 0.0 and not self.allow_pyramiding:
            return tuple(intents)

        quantity = float(self.quantity_policy.quantity(decision, quote, ledger))
        if quantity <= 0.0:
            return tuple(intents)
        intents.append(
            self._intent(
                decision,
                side=side,
                quantity=quantity,
                reduce_only=False,
                label="entry",
            )
        )
        return tuple(intents)

    def _exit_intents(
        self,
        decision: StrategyDecision,
        current: float,
    ) -> tuple[OrderIntent, ...]:
        if abs(current) <= 1e-12:
            return ()
        side = OrderSide.SELL if current > 0.0 else OrderSide.BUY
        return (
            self._intent(
                decision,
                side=side,
                quantity=abs(current),
                reduce_only=True,
                label="exit",
            ),
        )

    @staticmethod
    def _intent(
        decision: StrategyDecision,
        *,
        side: OrderSide,
        quantity: float,
        reduce_only: bool,
        label: str,
    ) -> OrderIntent:
        payload = (
            decision.strategy_id,
            decision.symbol,
            decision.as_of.isoformat(),
            decision.semantic_fingerprint,
            label,
            side.value,
            round(quantity, 12),
        )
        digest = sha256(repr(payload).encode("utf-8")).hexdigest()[:20]
        metadata: list[tuple[str, str]] = [
            ("decision_fingerprint", decision.semantic_fingerprint),
            ("signal_action", decision.action.value),
        ]
        if decision.stop_loss is not None:
            metadata.append(("stop_loss", repr(decision.stop_loss)))
        if decision.take_profit is not None:
            metadata.append(("take_profit", repr(decision.take_profit)))
        return OrderIntent(
            client_order_id=f"paper-{label}-{digest}",
            idempotency_key=f"paper:{digest}",
            symbol=decision.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            created_at=decision.as_of,
            time_in_force=TimeInForce.IOC,
            reduce_only=reduce_only,
            strategy_id=decision.strategy_id,
            metadata=tuple(metadata),
        )
