from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.models import SymbolSpec
from fxbot.risk.models import BrokerVolumeConstraints, InstrumentRiskSpec, TradeSide
from fxbot.risk.position_sizing import IdentityCurrencyConverter, StaticCurrencyConverter
from fxbot.risk.positions import (
    ManagedPosition,
    PositionLifecycle,
    PositionLifecycleError,
    PositionStatus,
)

NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


def instrument(*, quote: str = "USD") -> InstrumentRiskSpec:
    symbol = "EURUSD" if quote == "USD" else "EURGBP"
    return InstrumentRiskSpec(
        symbol=SymbolSpec(
            symbol=symbol,
            base_currency="EUR",
            quote_currency=quote,
            digits=5,
            point_size=0.00001,
            pip_size=0.0001,
            contract_size=100_000,
        ),
        volume=BrokerVolumeConstraints(0.01, 100, 0.01),
    )


def opened(
    *,
    side: TradeSide = TradeSide.LONG,
    volume: float = 1.0,
    entry: float = 1.10,
    stop: float | None = 1.095,
    take_profit: float | None = 1.12,
) -> ManagedPosition:
    if side is TradeSide.SHORT:
        stop = 1.105 if stop is not None else None
        take_profit = 1.08 if take_profit is not None else None
    return PositionLifecycle.open(
        position_id="p1",
        instrument=instrument(),
        side=side,
        volume=volume,
        entry_price=entry,
        opened_at=NOW,
        stop_price=stop,
        take_profit_price=take_profit,
        commission_account=2.0,
    )


def test_open_creates_versioned_open_position() -> None:
    position = opened()
    assert position.status is PositionStatus.OPEN
    assert position.total_opened_volume == 1.0
    assert position.open_volume == 1.0
    assert position.total_closed_volume == 0.0
    assert position.initial_stop_price == 1.095
    assert position.stop_price == 1.095
    assert position.version == 1
    assert position.net_realized_pnl_account == -2.0


def test_open_validates_long_and_short_protective_prices() -> None:
    with pytest.raises(ValueError, match="long position"):
        PositionLifecycle.open(
            position_id="p",
            instrument=instrument(),
            side=TradeSide.LONG,
            volume=1,
            entry_price=1.10,
            opened_at=NOW,
            stop_price=1.11,
        )
    with pytest.raises(ValueError, match="short position"):
        PositionLifecycle.open(
            position_id="p",
            instrument=instrument(),
            side=TradeSide.SHORT,
            volume=1,
            entry_price=1.10,
            opened_at=NOW,
            take_profit_price=1.11,
        )


def test_managed_position_rejects_inconsistent_volume_accounting() -> None:
    position = opened()
    with pytest.raises(ValueError, match="must equal"):
        ManagedPosition(
            position_id=position.position_id,
            instrument=position.instrument,
            side=position.side,
            total_opened_volume=1,
            open_volume=0.8,
            total_closed_volume=0.1,
            average_entry_price=position.average_entry_price,
            initial_stop_price=position.initial_stop_price,
            stop_price=position.stop_price,
            take_profit_price=position.take_profit_price,
            opened_at=NOW,
            updated_at=NOW,
        )


def test_add_entry_recalculates_weighted_average_and_version() -> None:
    position = opened(volume=1.0, take_profit=None)
    updated = PositionLifecycle.add_entry(
        position,
        volume=2.0,
        price=1.13,
        filled_at=NOW + timedelta(minutes=1),
        commission_account=1.5,
    )
    assert updated.open_volume == 3.0
    assert updated.total_opened_volume == 3.0
    assert updated.average_entry_price == pytest.approx(1.12)
    assert updated.commission_account == 3.5
    assert updated.version == 2


def test_add_entry_rejects_closed_position_and_stale_fill() -> None:
    closed = PositionLifecycle.reduce(
        opened(),
        volume=1,
        price=1.11,
        filled_at=NOW + timedelta(minutes=1),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    with pytest.raises(PositionLifecycleError, match="closed"):
        PositionLifecycle.add_entry(
            closed,
            volume=1,
            price=1.10,
            filled_at=NOW + timedelta(minutes=2),
        )
    with pytest.raises(PositionLifecycleError, match="predate"):
        PositionLifecycle.add_entry(
            opened(),
            volume=1,
            price=1.10,
            filled_at=NOW - timedelta(seconds=1),
        )


def test_add_entry_rejects_average_that_invalidates_initial_stop() -> None:
    position = opened(entry=1.10, stop=1.095, take_profit=None)
    with pytest.raises(ValueError, match="stop_price"):
        PositionLifecycle.add_entry(
            position,
            volume=10,
            price=1.09,
            filled_at=NOW + timedelta(minutes=1),
        )


def test_partial_exit_realizes_long_profit_and_updates_state() -> None:
    position = PositionLifecycle.reduce(
        opened(volume=2.0),
        volume=0.5,
        price=1.11,
        filled_at=NOW + timedelta(minutes=2),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
        commission_account=1.0,
        swap_account=-0.5,
    )
    assert position.status is PositionStatus.PARTIALLY_CLOSED
    assert position.open_volume == 1.5
    assert position.total_closed_volume == 0.5
    assert position.realized_pnl_account == pytest.approx(500.0)
    assert position.net_realized_pnl_account == pytest.approx(496.5)
    assert position.closed_at is None
    assert position.version == 2


def test_final_exit_closes_short_position_with_profit() -> None:
    position = PositionLifecycle.reduce(
        opened(side=TradeSide.SHORT),
        volume=1.0,
        price=1.08,
        filled_at=NOW + timedelta(minutes=3),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    assert position.status is PositionStatus.CLOSED
    assert position.open_volume == 0.0
    assert position.closed_at == NOW + timedelta(minutes=3)
    assert position.realized_pnl_account == pytest.approx(2_000.0)


def test_exit_converts_quote_pnl_to_account_currency() -> None:
    position = PositionLifecycle.open(
        position_id="cross",
        instrument=instrument(quote="GBP"),
        side=TradeSide.LONG,
        volume=1,
        entry_price=0.85,
        opened_at=NOW,
        stop_price=0.84,
    )
    closed = PositionLifecycle.reduce(
        position,
        volume=1,
        price=0.86,
        filled_at=NOW + timedelta(minutes=1),
        account_currency="USD",
        converter=StaticCurrencyConverter({("GBP", "USD"): 1.25}),
    )
    assert closed.realized_pnl_account == pytest.approx(1_250.0)


def test_reduce_rejects_over_close_stale_fill_and_closed_position() -> None:
    position = opened()
    with pytest.raises(PositionLifecycleError, match="exceeds"):
        PositionLifecycle.reduce(
            position,
            volume=1.1,
            price=1.11,
            filled_at=NOW + timedelta(minutes=1),
            account_currency="USD",
            converter=IdentityCurrencyConverter(),
        )
    with pytest.raises(PositionLifecycleError, match="predate"):
        PositionLifecycle.reduce(
            position,
            volume=0.1,
            price=1.11,
            filled_at=NOW - timedelta(seconds=1),
            account_currency="USD",
            converter=IdentityCurrencyConverter(),
        )
    closed = PositionLifecycle.reduce(
        position,
        volume=1,
        price=1.11,
        filled_at=NOW + timedelta(minutes=1),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    with pytest.raises(PositionLifecycleError, match="closed"):
        PositionLifecycle.reduce(
            closed,
            volume=0.1,
            price=1.11,
            filled_at=NOW + timedelta(minutes=2),
            account_currency="USD",
            converter=IdentityCurrencyConverter(),
        )


def test_mark_to_market_long_and_short() -> None:
    long_value = PositionLifecycle.value(
        opened(volume=0.5),
        market_price=1.11,
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    short_value = PositionLifecycle.value(
        opened(side=TradeSide.SHORT, volume=0.5),
        market_price=1.09,
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    assert long_value.unrealized_pnl_account == pytest.approx(500.0)
    assert short_value.unrealized_pnl_account == pytest.approx(500.0)
    assert long_value.total_pnl_account == pytest.approx(498.0)


def test_closed_position_has_zero_unrealized_pnl() -> None:
    closed = PositionLifecycle.reduce(
        opened(),
        volume=1,
        price=1.11,
        filled_at=NOW + timedelta(minutes=1),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    valuation = PositionLifecycle.value(
        closed,
        market_price=1.20,
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    assert valuation.unrealized_pnl_account == 0.0
    assert valuation.total_pnl_account == valuation.net_realized_pnl_account


def test_to_exposure_projects_open_state_and_rejects_closed() -> None:
    position = opened(volume=0.4)
    exposure = position.to_exposure(current_price=1.105, margin_used=440)
    assert exposure.position_id == "p1"
    assert exposure.volume == 0.4
    assert exposure.realized_pnl == -2.0
    closed = PositionLifecycle.reduce(
        position,
        volume=0.4,
        price=1.11,
        filled_at=NOW + timedelta(minutes=1),
        account_currency="USD",
        converter=IdentityCurrencyConverter(),
    )
    with pytest.raises(PositionLifecycleError, match="closed"):
        closed.to_exposure(current_price=1.11, margin_used=0)


def test_timestamps_must_be_aware_and_monotonic() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PositionLifecycle.open(
            position_id="p",
            instrument=instrument(),
            side=TradeSide.LONG,
            volume=1,
            entry_price=1.1,
            opened_at=datetime(2026, 1, 1),
            stop_price=1.09,
        )
