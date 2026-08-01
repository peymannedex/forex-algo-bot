from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from fxbot.data.adapters.mt5_adapter import MT5MarketDataAdapter
from fxbot.domain.enums import DataKind, Timeframe
from fxbot.domain.models import HistoricalDataRequest, LiveSubscription, Tick


class FakeMT5:
    COPY_TICKS_ALL = 0
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440
    TIMEFRAME_W1 = 10080

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False
        self.live_time_msc = 1_767_225_600_123

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        self.initialized = True
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def last_error(self) -> tuple[int, str]:
        return 0, "ok"

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def symbol_info(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(
            point=0.00001,
            digits=5,
            currency_base="EUR",
            currency_profit="USD",
            trade_contract_size=100_000.0,
        )

    def symbol_info_tick(self, symbol: str) -> dict[str, float | int]:
        return {
            "time": self.live_time_msc // 1000,
            "time_msc": self.live_time_msc,
            "bid": 1.1000,
            "ask": 1.1002,
        }

    def copy_ticks_range(
        self, symbol: str, date_from: datetime, date_to: datetime, flags: int
    ) -> list[dict[str, float | int]]:
        return [
            {
                "time": 1_767_225_600,
                "time_msc": 1_767_225_600_123,
                "bid": 1.1000,
                "ask": 1.1002,
            }
        ]

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime,
    ) -> list[dict[str, float | int]]:
        return [self._rate(1_767_225_600)]

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> list[dict[str, float | int]]:
        return [self._rate(1_767_225_540), self._rate(1_767_225_600)]

    @staticmethod
    def _rate(timestamp: int) -> dict[str, float | int]:
        return {
            "time": timestamp,
            "open": 1.1000,
            "high": 1.1010,
            "low": 1.0990,
            "close": 1.1005,
            "tick_volume": 42,
            "spread": 20,
            "real_volume": 0,
        }


def test_mt5_adapter_extracts_historical_ticks_and_bid_plus_spread_bars() -> None:
    client = FakeMT5()
    adapter = MT5MarketDataAdapter(client=client)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)

    ticks = list(
        adapter.iter_ticks(
            HistoricalDataRequest("EURUSD", DataKind.TICK, start=start, end=end)
        )
    )
    bars = list(
        adapter.iter_bars(
            HistoricalDataRequest(
                "EURUSD",
                DataKind.BAR,
                start=start,
                end=end,
                timeframe=Timeframe.M1,
            )
        )
    )
    adapter.close()

    assert len(ticks) == 1
    assert ticks[0].event_time.microsecond == 123_000
    assert len(bars) == 1
    assert bars[0].ask.open == pytest.approx(1.1002)
    assert bars[0].source == "mt5:bid-plus-spread"
    assert client.shutdown_called is True


@pytest.mark.asyncio
async def test_mt5_adapter_streams_and_deduplicates_live_tick() -> None:
    client = FakeMT5()
    adapter = MT5MarketDataAdapter(client=client, poll_interval_seconds=0.01)
    subscription = LiveSubscription(
        symbols=frozenset({"EURUSD"}),
        timeframes=frozenset({Timeframe.TICK}),
    )

    await adapter.connect()
    stream = adapter.stream(subscription)
    first = await anext(stream)
    await stream.aclose()
    await adapter.disconnect()

    assert isinstance(first, Tick)
    assert first.symbol == "EURUSD"
    assert client.shutdown_called is True
