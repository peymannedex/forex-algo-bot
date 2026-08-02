from datetime import UTC, datetime, timedelta

import pytest

from fxbot.backtest.broker import SimulatedBroker
from fxbot.backtest.config import (
    BacktestConfig,
    CommissionConfig,
    ExecutionConfig,
    InstrumentConfig,
    SlippageConfig,
    SwapConfig,
)
from fxbot.backtest.events import (
    MarketEvent,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from fxbot.domain.enums import Timeframe
from fxbot.domain.models import OHLC, Bar, Tick

BASE = datetime(2026, 1, 1, 20, 59, tzinfo=UTC)


def config(
    *,
    commission: float = 0.0,
    slippage_bps: float = 0.0,
    max_fill: float | None = None,
    max_spread_bps: float | None = None,
    swap_long: float = 0.0,
) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=10_000.0,
        instruments=(InstrumentConfig("EURUSD", contract_size=100_000.0, leverage=100),),
        commission=CommissionConfig(per_lot=commission),
        swap=SwapConfig(long_per_lot=swap_long, rollover_hour_utc=21),
        execution=ExecutionConfig(
            max_spread_bps=max_spread_bps,
            max_fill_volume_per_event=max_fill,
            slippage=SlippageConfig(base_bps=slippage_bps),
        ),
    )


def tick(sequence: int, bid: float, ask: float, *, at: datetime | None = None) -> MarketEvent:
    timestamp = at or BASE + timedelta(seconds=sequence)
    record = Tick(symbol="EURUSD", event_time=timestamp, bid=bid, ask=ask)
    return MarketEvent(sequence=sequence, timestamp=timestamp, record=record)


def request(
    order_id: str,
    sequence: int,
    *,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    volume: float = 1.0,
    limit_price: float | None = None,
    stop_price: float | None = None,
    reduce_only: bool = False,
) -> OrderRequest:
    return OrderRequest(
        order_id=order_id,
        symbol="EURUSD",
        side=side,
        order_type=order_type,
        volume=volume,
        submitted_at=BASE + timedelta(seconds=sequence),
        limit_price=limit_price,
        stop_price=stop_price,
        reduce_only=reduce_only,
    )


def test_market_order_fills_on_next_event_at_ask_with_costs() -> None:
    broker = SimulatedBroker(config(commission=3.0, slippage_bps=1.0))
    broker.on_market(tick(0, 1.1000, 1.1002))
    broker.submit(request("buy", 0), current_sequence=0)

    fills = broker.on_market(tick(1, 1.1010, 1.1012))
    assert len(fills) == 1
    expected = 1.1012 * 1.0001
    assert fills[0].price == pytest.approx(expected)
    assert fills[0].commission == pytest.approx(3.0)
    assert broker.orders[0].status is OrderStatus.FILLED
    snapshot = broker.snapshot(fills[0].timestamp)
    assert snapshot.positions[0].signed_volume == pytest.approx(1.0)
    assert snapshot.balance == pytest.approx(9_997.0)


def test_limit_and_stop_orders_use_side_specific_bar_prices() -> None:
    broker = SimulatedBroker(config())
    broker.on_market(tick(0, 1.1000, 1.1002))
    broker.submit(
        request(
            "buy-limit",
            0,
            order_type=OrderType.LIMIT,
            limit_price=1.0995,
        ),
        current_sequence=0,
    )
    broker.submit(
        request(
            "buy-stop",
            0,
            order_type=OrderType.STOP,
            stop_price=1.1020,
        ),
        current_sequence=0,
    )
    record = Bar(
        symbol="EURUSD",
        open_time=BASE + timedelta(seconds=1),
        timeframe=Timeframe.M1,
        bid=OHLC(1.1000, 1.1030, 1.0980, 1.1010),
        ask=OHLC(1.1002, 1.1032, 1.0982, 1.1012),
    )
    event = MarketEvent(sequence=1, timestamp=record.close_time, record=record)
    fills = broker.on_market(event)
    assert [item.order_id for item in fills] == ["buy-limit", "buy-stop"]
    assert fills[0].price == pytest.approx(1.0995)
    assert fills[1].price == pytest.approx(1.1020)


def test_partial_fills_respect_per_event_liquidity_cap() -> None:
    broker = SimulatedBroker(config(max_fill=0.4))
    broker.on_market(tick(0, 1.1, 1.1002))
    broker.submit(request("partial", 0, volume=1.0), current_sequence=0)
    assert broker.on_market(tick(1, 1.1, 1.1002))[0].volume == pytest.approx(0.4)
    assert broker.orders[0].status is OrderStatus.PARTIALLY_FILLED
    assert broker.on_market(tick(2, 1.1, 1.1002))[0].volume == pytest.approx(0.4)
    assert broker.on_market(tick(3, 1.1, 1.1002))[0].volume == pytest.approx(0.2)
    assert broker.orders[0].status is OrderStatus.FILLED


def test_spread_limit_rejects_order() -> None:
    broker = SimulatedBroker(config(max_spread_bps=2.0))
    broker.on_market(tick(0, 1.0, 1.0001))
    broker.submit(request("wide", 0), current_sequence=0)
    assert broker.on_market(tick(1, 1.0, 1.01)) == ()
    assert broker.orders[0].status is OrderStatus.REJECTED
    assert broker.orders[0].rejection_reason == "spread_limit_exceeded"


def test_reduce_only_close_realizes_bid_side_pnl() -> None:
    broker = SimulatedBroker(config())
    broker.on_market(tick(0, 1.1000, 1.1002))
    broker.submit(request("entry", 0), current_sequence=0)
    broker.on_market(tick(1, 1.1000, 1.1002))
    broker.submit(
        request(
            "exit",
            1,
            side=OrderSide.SELL,
            reduce_only=True,
        ),
        current_sequence=1,
    )
    broker.on_market(tick(2, 1.1010, 1.1012))
    snapshot = broker.snapshot(BASE + timedelta(seconds=2))
    assert snapshot.positions == ()
    assert snapshot.balance == pytest.approx(10_080.0)
    assert broker.trades[0].gross_pnl == pytest.approx(80.0)


def test_swap_is_applied_when_rollover_is_crossed() -> None:
    broker = SimulatedBroker(config(swap_long=-5.0))
    broker.on_market(tick(0, 1.1, 1.1002, at=BASE))
    broker.submit(
        OrderRequest(
            order_id="entry",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=1.0,
            submitted_at=BASE,
        ),
        current_sequence=0,
    )
    fill_time = BASE + timedelta(seconds=1)
    broker.on_market(tick(1, 1.1, 1.1002, at=fill_time))
    next_time = BASE + timedelta(minutes=2)
    broker.on_market(tick(2, 1.1, 1.1002, at=next_time))
    assert broker.snapshot(next_time).swap == pytest.approx(-5.0)


def test_seeded_slippage_is_reproducible() -> None:
    cfg = BacktestConfig(
        initial_cash=10_000,
        instruments=(InstrumentConfig("EURUSD"),),
        seed=99,
        execution=ExecutionConfig(slippage=SlippageConfig(base_bps=1, jitter_bps=1)),
    )
    prices: list[float] = []
    for _ in range(2):
        broker = SimulatedBroker(cfg)
        broker.on_market(tick(0, 1.1, 1.1002))
        broker.submit(request("order", 0), current_sequence=0)
        prices.append(broker.on_market(tick(1, 1.1, 1.1002))[0].price)
    assert prices[0] == prices[1]
