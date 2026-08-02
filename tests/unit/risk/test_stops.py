from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.models import SymbolSpec
from fxbot.risk.models import BrokerVolumeConstraints, InstrumentRiskSpec, TradeSide
from fxbot.risk.position_sizing import IdentityCurrencyConverter
from fxbot.risk.positions import PositionLifecycle, PositionLifecycleError
from fxbot.risk.stops import (
    StopManagementPolicy,
    StopManager,
    StopUpdateReason,
)

NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


def instrument() -> InstrumentRiskSpec:
    return InstrumentRiskSpec(
        symbol=SymbolSpec("EURUSD", "EUR", "USD", 5, 0.00001, 0.0001),
        volume=BrokerVolumeConstraints(0.01, 100, 0.01),
    )


def position(side: TradeSide = TradeSide.LONG):
    return PositionLifecycle.open(
        position_id="p1",
        instrument=instrument(),
        side=side,
        volume=1,
        entry_price=1.10,
        opened_at=NOW,
        stop_price=1.095 if side is TradeSide.LONG else 1.105,
    )


def test_policy_requires_trailing_distance_when_enabled() -> None:
    with pytest.raises(ValueError, match="trailing_distance"):
        StopManagementPolicy(trailing_activation_r=1.0, trailing_distance=None)


def test_favorable_r_multiple_for_long_and_short() -> None:
    manager = StopManager()
    assert manager.favorable_r_multiple(position(), 1.11) == pytest.approx(2.0)
    assert manager.favorable_r_multiple(position(TradeSide.SHORT), 1.09) == pytest.approx(2.0)


def test_favorable_r_is_none_without_initial_stop() -> None:
    no_stop = PositionLifecycle.open(
        position_id="p",
        instrument=instrument(),
        side=TradeSide.LONG,
        volume=1,
        entry_price=1.10,
        opened_at=NOW,
    )
    assert StopManager().favorable_r_multiple(no_stop, 1.11) is None


def test_no_change_below_activation_thresholds() -> None:
    update = StopManager().evaluate(
        position(),
        market_price=1.102,
        policy=StopManagementPolicy(trailing_distance=0.002),
    )
    assert not update.changed
    assert update.reason is StopUpdateReason.NONE
    assert update.applied_stop_price == 1.095


def test_break_even_update_for_long() -> None:
    update = StopManager().evaluate(
        position(),
        market_price=1.106,
        policy=StopManagementPolicy(
            break_even_trigger_r=1.0,
            break_even_offset=0.0002,
            trailing_activation_r=None,
        ),
    )
    assert update.changed
    assert update.reason is StopUpdateReason.BREAK_EVEN
    assert update.applied_stop_price == pytest.approx(1.1002)


def test_break_even_update_for_short() -> None:
    update = StopManager().evaluate(
        position(TradeSide.SHORT),
        market_price=1.094,
        policy=StopManagementPolicy(
            break_even_trigger_r=1.0,
            break_even_offset=0.0002,
            trailing_activation_r=None,
        ),
    )
    assert update.changed
    assert update.applied_stop_price == pytest.approx(1.0998)


def test_trailing_selects_most_protective_candidate() -> None:
    update = StopManager().evaluate(
        position(),
        market_price=1.112,
        policy=StopManagementPolicy(
            break_even_trigger_r=1.0,
            trailing_activation_r=1.5,
            trailing_distance=0.003,
        ),
    )
    assert update.reason is StopUpdateReason.BREAK_EVEN_AND_TRAILING
    assert update.applied_stop_price == pytest.approx(1.109)


def test_short_trailing_selects_lower_stop() -> None:
    update = StopManager().evaluate(
        position(TradeSide.SHORT),
        market_price=1.088,
        policy=StopManagementPolicy(
            break_even_trigger_r=1.0,
            trailing_activation_r=1.5,
            trailing_distance=0.003,
        ),
    )
    assert update.applied_stop_price == pytest.approx(1.091)


def test_tighten_only_rejects_widening_long_and_short() -> None:
    manager = StopManager()
    long_update = manager.manual_update(
        position(),
        market_price=1.10,
        stop_price=1.094,
    )
    short_update = manager.manual_update(
        position(TradeSide.SHORT),
        market_price=1.10,
        stop_price=1.106,
    )
    assert not long_update.changed
    assert "tighten" in (long_update.rejection_reason or "")
    assert not short_update.changed


def test_minimum_market_distance_rejects_too_close_stop() -> None:
    update = StopManager().manual_update(
        position(),
        market_price=1.105,
        stop_price=1.1045,
        policy=StopManagementPolicy(
            break_even_trigger_r=None,
            trailing_activation_r=None,
            minimum_market_distance=0.001,
        ),
    )
    assert not update.changed
    assert "minimum distance" in (update.rejection_reason or "")


def test_trailing_step_filters_small_improvement() -> None:
    update = StopManager().manual_update(
        position(),
        market_price=1.105,
        stop_price=1.0955,
        policy=StopManagementPolicy(
            break_even_trigger_r=None,
            trailing_activation_r=None,
            trailing_step=0.001,
        ),
    )
    assert not update.changed
    assert "trailing step" in (update.rejection_reason or "")


def test_apply_updates_position_immutably_and_increments_version() -> None:
    original = position()
    update = StopManager().manual_update(
        original,
        market_price=1.105,
        stop_price=1.10,
    )
    amended = StopManager.apply(
        original,
        update,
        updated_at=NOW + timedelta(minutes=1),
    )
    assert original.stop_price == 1.095
    assert amended.stop_price == 1.10
    assert amended.version == 2


def test_apply_noop_returns_same_object() -> None:
    original = position()
    update = StopManager().evaluate(
        original,
        market_price=1.101,
        policy=StopManagementPolicy(trailing_distance=0.002),
    )
    assert StopManager.apply(original, update, updated_at=NOW) is original


def test_apply_rejects_wrong_position_stale_time_and_closed_position() -> None:
    original = position()
    update = StopManager().manual_update(original, market_price=1.105, stop_price=1.10)
    wrong = PositionLifecycle.open(
        position_id="other",
        instrument=instrument(),
        side=TradeSide.LONG,
        volume=1,
        entry_price=1.10,
        opened_at=NOW,
        stop_price=1.095,
    )
    with pytest.raises(PositionLifecycleError, match="different"):
        StopManager.apply(wrong, update, updated_at=NOW + timedelta(minutes=1))
    with pytest.raises(PositionLifecycleError, match="predate"):
        StopManager.apply(original, update, updated_at=NOW - timedelta(seconds=1))
    closed = PositionLifecycle.reduce(
        original,
        volume=1,
        price=1.11,
        filled_at=NOW + timedelta(minutes=1),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    with pytest.raises(PositionLifecycleError, match="closed"):
        StopManager().manual_update(closed, market_price=1.11, stop_price=1.10)
