from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from fxbot.data.adapters.mt5_adapter import MT5AdapterError
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar, Tick
from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.integration.live_feed import MT5ReadOnlyMarketSource
from fxbot.production.config import ProductionSettings


def production_settings() -> ProductionSettings:
    return ProductionSettings(
        profile="paper",
        mt5_login=123,
        mt5_password=SecretStr("secret"),
        mt5_server="Demo",
    )


def test_mt5_source_is_market_data_only() -> None:
    source = MT5ReadOnlyMarketSource(
        production_settings(),
        PaperLiveFeedSettings(),
    )

    assert hasattr(source, "connect")
    assert hasattr(source, "stream")
    assert not hasattr(source, "submit_order")


def test_source_normalizes_tick_and_bar_server_time_to_utc() -> None:
    now = datetime(2026, 8, 4, 1, 48, 52, tzinfo=UTC)
    source = MT5ReadOnlyMarketSource(
        production_settings(),
        PaperLiveFeedSettings(
            mt5_server_utc_offset_minutes=180,
        ),
        clock=lambda: now,
    )
    raw_tick = Tick(
        symbol="EURUSD",
        event_time=now + timedelta(hours=3),
        bid=1.1000,
        ask=1.1002,
    )
    raw_bar = Bar(
        symbol="EURUSD",
        open_time=datetime(2026, 8, 4, 4, 40, tzinfo=UTC),
        timeframe=Timeframe.M5,
        bid=OHLC(1.1000, 1.1010, 1.0990, 1.1005),
        ask=OHLC(1.1002, 1.1012, 1.0992, 1.1007),
        complete=True,
    )

    tick = source._normalize_record(raw_tick)
    bar = source._normalize_record(raw_bar)

    assert isinstance(tick, Tick)
    assert tick.event_time == now
    assert isinstance(bar, Bar)
    assert bar.open_time == datetime(2026, 8, 4, 1, 40, tzinfo=UTC)
    assert bar.close_time == datetime(2026, 8, 4, 1, 45, tzinfo=UTC)

    source._validate_live_timestamp(tick)
    source._validate_live_timestamp(bar)


def test_source_rejects_record_still_in_the_future() -> None:
    now = datetime(2026, 8, 4, 1, 48, 52, tzinfo=UTC)
    source = MT5ReadOnlyMarketSource(
        production_settings(),
        PaperLiveFeedSettings(
            mt5_server_utc_offset_minutes=0,
            max_future_skew_seconds=5.0,
        ),
        clock=lambda: now,
    )
    future_tick = Tick(
        symbol="EURUSD",
        event_time=now + timedelta(hours=3),
        bid=1.1000,
        ask=1.1002,
    )

    with pytest.raises(MT5AdapterError, match="future-dated"):
        source._validate_live_timestamp(future_tick)
