from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxbot.domain.models import SymbolSpec
from fxbot.risk.models import (
    AccountSnapshot,
    BrokerVolumeConstraints,
    InstrumentRiskSpec,
    TradeSide,
)
from fxbot.risk.portfolio import (
    PendingOrderExposure,
    PortfolioAnalyzer,
    PortfolioSnapshot,
    PositionExposure,
    TradeProposal,
)
from fxbot.risk.position_sizing import StaticCurrencyConverter

NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


def eurusd() -> InstrumentRiskSpec:
    return InstrumentRiskSpec(
        symbol=SymbolSpec(
            symbol="EURUSD",
            base_currency="EUR",
            quote_currency="USD",
            digits=5,
            point_size=0.00001,
            pip_size=0.0001,
            contract_size=100_000,
        ),
        volume=BrokerVolumeConstraints(0.01, 100, 0.01),
    )


def account(
    *,
    balance: float = 10_000,
    equity: float = 10_000,
    margin_used: float = 0,
) -> AccountSnapshot:
    return AccountSnapshot(
        currency="USD",
        balance=balance,
        equity=equity,
        free_margin=max(equity - margin_used, 0),
        margin_used=margin_used,
        leverage=100,
    )


def converter() -> StaticCurrencyConverter:
    return StaticCurrencyConverter({("EUR", "USD"): 1.10})


def position(
    *,
    position_id: str = "p1",
    side: TradeSide = TradeSide.LONG,
    volume: float = 1.0,
    entry: float = 1.10,
    current: float = 1.105,
    stop: float | None = 1.095,
    margin: float = 1_100,
) -> PositionExposure:
    return PositionExposure(
        position_id=position_id,
        instrument=eurusd(),
        side=side,
        volume=volume,
        entry_price=entry,
        current_price=current,
        stop_price=stop,
        margin_used=margin,
    )


def pending(
    *,
    order_id: str = "o1",
    side: TradeSide = TradeSide.LONG,
    volume: float = 0.5,
    entry: float = 1.10,
    stop: float | None = 1.095,
    margin: float = 550,
) -> PendingOrderExposure:
    return PendingOrderExposure(
        order_id=order_id,
        instrument=eurusd(),
        side=side,
        volume=volume,
        entry_price=entry,
        stop_price=stop,
        margin_required=margin,
    )


def snapshot(
    *,
    positions: tuple[PositionExposure, ...] = (),
    pending_orders: tuple[PendingOrderExposure, ...] = (),
    equity: float = 10_000,
    balance: float = 10_000,
    margin_used: float = 0,
    day_start_equity: float = 10_000,
    peak: float = 10_000,
    realized: float = 0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        account=account(balance=balance, equity=equity, margin_used=margin_used),
        as_of=NOW,
        day_start_equity=day_start_equity,
        intraday_peak_equity=peak,
        realized_pnl_today=realized,
        positions=positions,
        pending_orders=pending_orders,
    )


def test_position_validates_directional_stop_and_reports_unrealized_pnl() -> None:
    item = position()
    assert item.symbol == "EURUSD"
    assert item.signed_volume == 1.0
    assert item.unrealized_pnl_quote == pytest.approx(500)
    with pytest.raises(ValueError, match="long exposure"):
        position(stop=1.11)


def test_short_position_reports_negative_signed_volume_and_profit() -> None:
    item = position(
        side=TradeSide.SHORT,
        entry=1.10,
        current=1.09,
        stop=1.105,
    )
    assert item.signed_volume == -1.0
    assert item.unrealized_pnl_quote == pytest.approx(1_000)


def test_pending_order_validates_values() -> None:
    item = pending()
    assert item.symbol == "EURUSD"
    with pytest.raises(ValueError, match="short exposure"):
        pending(side=TradeSide.SHORT, stop=1.09)


def test_trade_proposal_converts_to_position_and_pending_order() -> None:
    proposal = TradeProposal(
        proposal_id="trade-1",
        instrument=eurusd(),
        side=TradeSide.LONG,
        volume=0.2,
        entry_price=1.10,
        stop_price=1.095,
        margin_required=220,
        submitted_at=NOW,
    )
    assert proposal.as_position().position_id == "proposal:trade-1"
    assert proposal.as_pending_order().order_id == "proposal:trade-1"


def test_snapshot_normalizes_times_and_rejects_duplicate_ids() -> None:
    item = snapshot(positions=(position(position_id="same"),))
    assert item.as_of.tzinfo is UTC
    with pytest.raises(ValueError, match="identifiers"):
        snapshot(
            positions=(position(position_id="same"),),
            pending_orders=(pending(order_id="same"),),
        )


def test_snapshot_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PortfolioSnapshot(
            account=account(),
            as_of=datetime(2026, 8, 2, 12),
            day_start_equity=10_000,
            intraday_peak_equity=10_000,
        )


def test_analyzer_aggregates_risk_notional_margin_and_pnl() -> None:
    item = snapshot(
        positions=(position(),),
        equity=9_800,
        margin_used=1_100,
        day_start_equity=10_000,
        peak=10_100,
        realized=-100,
    )
    metrics = PortfolioAnalyzer(converter()).analyze(item)
    assert metrics.open_position_count == 1
    assert metrics.open_risk_amount == pytest.approx(500)
    assert metrics.total_risk_amount == pytest.approx(500)
    assert metrics.gross_notional_amount == pytest.approx(110_500)
    assert metrics.net_notional_amount == pytest.approx(110_500)
    assert metrics.margin_committed == pytest.approx(1_100)
    assert metrics.margin_utilization == pytest.approx(1_100 / 9_800)
    assert metrics.unrealized_pnl == pytest.approx(500)
    assert metrics.daily_realized_loss_amount == pytest.approx(100)
    assert metrics.daily_total_loss_amount == pytest.approx(200)
    assert metrics.intraday_drawdown_amount == pytest.approx(300)
    assert metrics.symbol_count("eurusd") == 1
    assert metrics.currency_exposure("EUR") == pytest.approx(110_000)
    assert metrics.currency_exposure("USD") == pytest.approx(-110_500)


def test_analyzer_uses_account_margin_when_larger_than_position_sum() -> None:
    item = snapshot(
        positions=(position(margin=500),),
        margin_used=900,
    )
    metrics = PortfolioAnalyzer(converter()).analyze(item)
    assert metrics.margin_committed == 900


def test_analyzer_additional_position_adds_margin_once() -> None:
    item = snapshot(positions=(position(margin=500),), margin_used=500)
    extra = position(position_id="p2", volume=0.1, margin=110)
    metrics = PortfolioAnalyzer(converter()).analyze(
        item,
        additional_positions=(extra,),
    )
    assert metrics.open_position_count == 2
    assert metrics.margin_committed == pytest.approx(610)


def test_offsetting_positions_reduce_net_but_not_gross_notional() -> None:
    long = position(position_id="long", volume=1, current=1.10)
    short = position(
        position_id="short",
        side=TradeSide.SHORT,
        volume=1,
        current=1.10,
        stop=1.105,
    )
    metrics = PortfolioAnalyzer(converter()).analyze(
        snapshot(positions=(long, short), margin_used=2_200)
    )
    assert metrics.gross_notional_amount == pytest.approx(220_000)
    assert metrics.net_notional_amount == pytest.approx(0)


def test_pending_exposure_can_be_excluded_from_monetary_metrics() -> None:
    item = snapshot(pending_orders=(pending(),))
    analyzer = PortfolioAnalyzer(converter())
    included = analyzer.analyze(item, include_pending_orders=True)
    excluded = analyzer.analyze(item, include_pending_orders=False)
    assert included.pending_order_count == excluded.pending_order_count == 1
    assert included.pending_risk_amount == pytest.approx(250)
    assert included.gross_notional_amount > 0
    assert excluded.pending_risk_amount == 0
    assert excluded.gross_notional_amount == 0
    assert excluded.margin_committed == 0


def test_unprotected_exposures_are_counted() -> None:
    metrics = PortfolioAnalyzer(converter()).analyze(
        snapshot(
            positions=(position(stop=None),),
            pending_orders=(pending(stop=None),),
        )
    )
    assert metrics.unprotected_exposure_count == 2
    assert metrics.total_risk_amount == 0


def test_metrics_helpers_return_zero_for_unknown_scope() -> None:
    metrics = PortfolioAnalyzer(converter()).analyze(snapshot())
    assert metrics.symbol_count("GBPUSD") == 0
    assert metrics.symbol_notional("GBPUSD") == 0
    assert metrics.currency_exposure("GBP") == 0
    assert metrics.largest_currency_exposure == 0
