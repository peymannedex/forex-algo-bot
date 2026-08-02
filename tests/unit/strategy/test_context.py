from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar
from fxbot.strategy.context import (
    ContextIssueCode,
    MarketContextBuilder,
    MarketSeries,
    MultiTimeframeContext,
    timeframe_age_tolerance,
)
from fxbot.strategy.models import StrategyConfig

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_bar(
    index: int,
    *,
    timeframe: Timeframe = Timeframe.M5,
    symbol: str = "EURUSD",
    complete: bool = True,
) -> Bar:
    seconds = timeframe.seconds
    assert seconds is not None
    close = 1.10 + index * 0.0001
    spread = 0.0002
    bid = OHLC(close - spread, close + 0.0002 - spread, close - 0.0002 - spread, close - spread)
    ask = OHLC(close, close + 0.0002, close - 0.0002, close)
    return Bar(
        symbol=symbol,
        open_time=BASE + timedelta(seconds=seconds * index),
        timeframe=timeframe,
        bid=bid,
        ask=ask,
        complete=complete,
    )


def test_market_series_exposes_prices_and_window() -> None:
    series = MarketSeries("EURUSD", Timeframe.M5, tuple(make_bar(i) for i in range(5)))
    assert len(series.closes) == 5
    assert len(series.window(2).bars) == 2
    assert series.latest.open_time == make_bar(4).open_time


def test_market_series_rejects_out_of_order_bars() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketSeries("EURUSD", Timeframe.M5, (make_bar(1), make_bar(0)))


def test_market_series_rejects_symbol_and_timeframe_mismatch() -> None:
    with pytest.raises(ValueError, match="symbol"):
        MarketSeries("EURUSD", Timeframe.M5, (make_bar(0, symbol="GBPUSD"),))
    with pytest.raises(ValueError, match="timeframe"):
        MarketSeries("EURUSD", Timeframe.M5, (make_bar(0, timeframe=Timeframe.H1),))


def test_available_at_prevents_complete_bar_lookahead() -> None:
    series = MarketSeries("EURUSD", Timeframe.M5, tuple(make_bar(i) for i in range(3)))
    cutoff = make_bar(1).close_time
    available = series.available_at(cutoff)
    assert len(available.bars) == 2


def test_context_requires_primary_and_unique_timeframes() -> None:
    m5 = MarketSeries("EURUSD", Timeframe.M5, (make_bar(0),))
    with pytest.raises(ValueError, match="Primary"):
        MultiTimeframeContext("EURUSD", BASE, Timeframe.H1, (m5,))
    with pytest.raises(ValueError, match="Duplicate"):
        MultiTimeframeContext("EURUSD", BASE, Timeframe.M5, (m5, m5))


def test_context_get_and_timeframes() -> None:
    m5 = MarketSeries("EURUSD", Timeframe.M5, (make_bar(0),))
    h1 = MarketSeries(
        "EURUSD",
        Timeframe.H1,
        (make_bar(0, timeframe=Timeframe.H1),),
    )
    context = MultiTimeframeContext("EURUSD", BASE + timedelta(hours=1), Timeframe.M5, (m5, h1))
    assert context.primary is m5
    assert set(context.timeframes) == {Timeframe.M5, Timeframe.H1}
    with pytest.raises(KeyError):
        context.get(Timeframe.H4)


def test_context_validation_reports_missing_and_warmup() -> None:
    bars = tuple(make_bar(i) for i in range(3))
    context = MultiTimeframeContext(
        "EURUSD",
        bars[-1].close_time,
        Timeframe.M5,
        (MarketSeries("EURUSD", Timeframe.M5, bars),),
    )
    config = StrategyConfig(
        strategy_id="s",
        primary_timeframe=Timeframe.M5,
        required_timeframes=(Timeframe.H1,),
        warmup_bars=5,
        max_data_age=timedelta(minutes=10),
    )
    codes = {issue.code for issue in context.validate(config)}
    assert ContextIssueCode.INSUFFICIENT_WARMUP in codes
    assert ContextIssueCode.MISSING_TIMEFRAME in codes


def test_context_validation_reports_incomplete_future_and_stale() -> None:
    incomplete = make_bar(0, complete=False)
    future = make_bar(1)
    context = MultiTimeframeContext(
        "EURUSD",
        BASE,
        Timeframe.M5,
        (MarketSeries("EURUSD", Timeframe.M5, (incomplete, future)),),
    )
    config = StrategyConfig(
        strategy_id="s",
        primary_timeframe=Timeframe.M5,
        warmup_bars=1,
        max_data_age=timedelta(seconds=1),
    )
    codes = {issue.code for issue in context.validate(config)}
    assert ContextIssueCode.INCOMPLETE_BAR in codes
    assert ContextIssueCode.FUTURE_DATA in codes


def test_context_validation_reports_stale_primary() -> None:
    bar = make_bar(0)
    context = MultiTimeframeContext(
        "EURUSD",
        bar.close_time + timedelta(hours=1),
        Timeframe.M5,
        (MarketSeries("EURUSD", Timeframe.M5, (bar,)),),
    )
    config = StrategyConfig(
        strategy_id="s",
        primary_timeframe=Timeframe.M5,
        warmup_bars=1,
        max_data_age=timedelta(minutes=5),
    )
    assert context.validate(config)[0].code is ContextIssueCode.STALE_PRIMARY_DATA


def test_builder_groups_and_excludes_future_data() -> None:
    bars = [
        make_bar(0),
        make_bar(1),
        make_bar(0, timeframe=Timeframe.H1),
    ]
    context = MarketContextBuilder().build(
        symbol="EURUSD",
        as_of=make_bar(0).close_time,
        primary_timeframe=Timeframe.M5,
        bars=bars,
    )
    assert len(context.get(Timeframe.M5).bars) == 1
    assert Timeframe.H1 not in context.timeframes


def test_timeframe_age_tolerance() -> None:
    assert timeframe_age_tolerance(Timeframe.M5, timedelta(seconds=30)) == timedelta(
        minutes=5,
        seconds=30,
    )
    with pytest.raises(ValueError):
        timeframe_age_tolerance(Timeframe.TICK, timedelta(0))
