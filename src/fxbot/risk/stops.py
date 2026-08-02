"""Protective-stop, break-even, and trailing-stop management."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from math import isfinite

from fxbot.risk.models import TradeSide
from fxbot.risk.positions import ManagedPosition, PositionLifecycleError, PositionStatus


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


class StopUpdateReason(StrEnum):
    """Why the stop engine proposed or applied a change."""

    NONE = "none"
    MANUAL = "manual"
    BREAK_EVEN = "break_even"
    TRAILING = "trailing"
    BREAK_EVEN_AND_TRAILING = "break_even_and_trailing"


@dataclass(frozen=True, slots=True)
class StopManagementPolicy:
    """Deterministic break-even and trailing-stop configuration."""

    break_even_trigger_r: float | None = 1.0
    break_even_offset: float = 0.0
    trailing_activation_r: float | None = 1.5
    trailing_distance: float | None = None
    trailing_step: float = 0.0
    minimum_market_distance: float = 0.0
    tighten_only: bool = True

    def __post_init__(self) -> None:
        if self.break_even_trigger_r is not None:
            object.__setattr__(
                self,
                "break_even_trigger_r",
                _positive(self.break_even_trigger_r, "break_even_trigger_r"),
            )
        object.__setattr__(
            self,
            "break_even_offset",
            _non_negative(self.break_even_offset, "break_even_offset"),
        )
        if self.trailing_activation_r is not None:
            object.__setattr__(
                self,
                "trailing_activation_r",
                _positive(self.trailing_activation_r, "trailing_activation_r"),
            )
        if self.trailing_distance is not None:
            object.__setattr__(
                self,
                "trailing_distance",
                _positive(self.trailing_distance, "trailing_distance"),
            )
        object.__setattr__(
            self,
            "trailing_step",
            _non_negative(self.trailing_step, "trailing_step"),
        )
        object.__setattr__(
            self,
            "minimum_market_distance",
            _non_negative(self.minimum_market_distance, "minimum_market_distance"),
        )
        if self.trailing_activation_r is not None and self.trailing_distance is None:
            raise ValueError("trailing_distance is required when trailing activation is enabled")


@dataclass(frozen=True, slots=True)
class StopUpdate:
    """Auditable output of one stop-management evaluation."""

    position_id: str
    reason: StopUpdateReason
    market_price: float
    old_stop_price: float | None
    proposed_stop_price: float | None
    applied_stop_price: float | None
    changed: bool
    favorable_r_multiple: float | None
    rejection_reason: str | None = None


class StopManager:
    """Evaluate and immutably apply protective-stop updates."""

    @staticmethod
    def favorable_r_multiple(position: ManagedPosition, market_price: float) -> float | None:
        """Return favorable move divided by initial stop risk distance."""

        market = _positive(market_price, "market_price")
        risk = position.initial_risk_distance
        if risk is None or risk <= 0.0:
            return None
        favorable_move = (
            market - position.average_entry_price
            if position.side is TradeSide.LONG
            else position.average_entry_price - market
        )
        return favorable_move / risk

    def evaluate(
        self,
        position: ManagedPosition,
        *,
        market_price: float,
        policy: StopManagementPolicy,
    ) -> StopUpdate:
        """Evaluate break-even and trailing candidates without mutating state."""

        market = _positive(market_price, "market_price")
        self._require_open(position)
        r_multiple = self.favorable_r_multiple(position, market)
        candidates: list[tuple[float, StopUpdateReason]] = []

        if (
            policy.break_even_trigger_r is not None
            and r_multiple is not None
            and r_multiple >= policy.break_even_trigger_r
        ):
            break_even = (
                position.average_entry_price + policy.break_even_offset
                if position.side is TradeSide.LONG
                else position.average_entry_price - policy.break_even_offset
            )
            candidates.append((break_even, StopUpdateReason.BREAK_EVEN))

        if (
            policy.trailing_activation_r is not None
            and policy.trailing_distance is not None
            and r_multiple is not None
            and r_multiple >= policy.trailing_activation_r
        ):
            trailing = (
                market - policy.trailing_distance
                if position.side is TradeSide.LONG
                else market + policy.trailing_distance
            )
            candidates.append((trailing, StopUpdateReason.TRAILING))

        if not candidates:
            return StopUpdate(
                position_id=position.position_id,
                reason=StopUpdateReason.NONE,
                market_price=market,
                old_stop_price=position.stop_price,
                proposed_stop_price=None,
                applied_stop_price=position.stop_price,
                changed=False,
                favorable_r_multiple=r_multiple,
            )

        proposed = self._most_protective(position.side, candidates)
        reasons = {reason for _, reason in candidates}
        reason = (
            StopUpdateReason.BREAK_EVEN_AND_TRAILING
            if len(reasons) > 1
            else next(iter(reasons))
        )
        return self._validate_candidate(
            position,
            market_price=market,
            candidate=proposed,
            reason=reason,
            favorable_r_multiple=r_multiple,
            policy=policy,
        )

    def manual_update(
        self,
        position: ManagedPosition,
        *,
        market_price: float,
        stop_price: float,
        policy: StopManagementPolicy | None = None,
    ) -> StopUpdate:
        """Validate a manually requested stop change under the same safety rules."""

        market = _positive(market_price, "market_price")
        candidate = _positive(stop_price, "stop_price")
        self._require_open(position)
        return self._validate_candidate(
            position,
            market_price=market,
            candidate=candidate,
            reason=StopUpdateReason.MANUAL,
            favorable_r_multiple=self.favorable_r_multiple(position, market),
            policy=policy or StopManagementPolicy(
                break_even_trigger_r=None,
                trailing_activation_r=None,
            ),
        )

    @staticmethod
    def apply(
        position: ManagedPosition,
        update: StopUpdate,
        *,
        updated_at: datetime,
    ) -> ManagedPosition:
        """Apply an accepted update and increment the position version."""

        if update.position_id != position.position_id:
            raise PositionLifecycleError("Stop update belongs to a different position")
        if not update.changed or update.applied_stop_price is None:
            return position
        if position.status is PositionStatus.CLOSED:
            raise PositionLifecycleError("Cannot update the stop of a closed position")
        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        if updated_at < position.updated_at:
            raise PositionLifecycleError("Stop update cannot predate position state")
        return replace(
            position,
            stop_price=update.applied_stop_price,
            updated_at=updated_at,
            version=position.version + 1,
        )

    @staticmethod
    def _require_open(position: ManagedPosition) -> None:
        if position.status is PositionStatus.CLOSED:
            raise PositionLifecycleError("Cannot manage stops for a closed position")

    @staticmethod
    def _most_protective(
        side: TradeSide,
        candidates: list[tuple[float, StopUpdateReason]],
    ) -> float:
        prices = [price for price, _ in candidates]
        return max(prices) if side is TradeSide.LONG else min(prices)

    @staticmethod
    def _validate_candidate(
        position: ManagedPosition,
        *,
        market_price: float,
        candidate: float,
        reason: StopUpdateReason,
        favorable_r_multiple: float | None,
        policy: StopManagementPolicy,
    ) -> StopUpdate:
        old = position.stop_price
        minimum = policy.minimum_market_distance
        rejection: str | None = None

        if position.side is TradeSide.LONG:
            if candidate >= market_price - minimum:
                rejection = "Long stop must remain below market by the minimum distance"
            elif policy.tighten_only and old is not None and candidate <= old:
                rejection = "Long stop update would not tighten protection"
            elif old is not None and candidate - old < policy.trailing_step:
                rejection = "Stop improvement is below the configured trailing step"
        else:
            if candidate <= market_price + minimum:
                rejection = "Short stop must remain above market by the minimum distance"
            elif policy.tighten_only and old is not None and candidate >= old:
                rejection = "Short stop update would not tighten protection"
            elif old is not None and old - candidate < policy.trailing_step:
                rejection = "Stop improvement is below the configured trailing step"

        return StopUpdate(
            position_id=position.position_id,
            reason=reason,
            market_price=market_price,
            old_stop_price=old,
            proposed_stop_price=candidate,
            applied_stop_price=candidate if rejection is None else old,
            changed=rejection is None and candidate != old,
            favorable_r_multiple=favorable_r_multiple,
            rejection_reason=rejection,
        )
