from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fxbot.domain.models import SymbolSpec
from fxbot.risk.limits import PortfolioRiskLimits, RiskLimitCode
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
from fxbot.risk.risk_guard import RiskDecisionStatus, RiskGuard

NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


def instrument() -> InstrumentRiskSpec:
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


def position(
    position_id: str,
    *,
    side: TradeSide = TradeSide.LONG,
    volume: float = 0.1,
    stop: float | None = 1.095,
    margin: float = 110,
) -> PositionExposure:
    return PositionExposure(
        position_id=position_id,
        instrument=instrument(),
        side=side,
        volume=volume,
        entry_price=1.10,
        current_price=1.10,
        stop_price=stop if side is TradeSide.LONG else 1.105,
        margin_used=margin,
    )


def pending(order_id: str, *, volume: float = 0.1) -> PendingOrderExposure:
    return PendingOrderExposure(
        order_id=order_id,
        instrument=instrument(),
        side=TradeSide.LONG,
        volume=volume,
        entry_price=1.10,
        stop_price=1.095,
        margin_required=110 * volume / 0.1,
    )


def snapshot(
    *,
    positions: tuple[PositionExposure, ...] = (),
    pending_orders: tuple[PendingOrderExposure, ...] = (),
    balance: float = 10_000,
    equity: float = 10_000,
    day_start: float = 10_000,
    peak: float = 10_000,
    realized: float = 0,
    consecutive_losses: int = 0,
    last_loss_time: datetime | None = None,
    manual_kill: bool = False,
    automatic_kill: bool = False,
) -> PortfolioSnapshot:
    margin = sum(item.margin_used for item in positions)
    return PortfolioSnapshot(
        account=account(balance=balance, equity=equity, margin_used=margin),
        as_of=NOW,
        day_start_equity=day_start,
        intraday_peak_equity=peak,
        realized_pnl_today=realized,
        consecutive_losses=consecutive_losses,
        last_loss_time=last_loss_time,
        manual_kill_switch=manual_kill,
        automatic_kill_switch=automatic_kill,
        kill_switch_reason="test" if manual_kill or automatic_kill else None,
        positions=positions,
        pending_orders=pending_orders,
    )


def proposal(
    *,
    proposal_id: str = "new",
    side: TradeSide = TradeSide.LONG,
    volume: float = 0.1,
    stop: float | None = 1.095,
    margin: float = 110,
    pending_order: bool = False,
) -> TradeProposal:
    return TradeProposal(
        proposal_id=proposal_id,
        instrument=instrument(),
        side=side,
        volume=volume,
        entry_price=1.10,
        stop_price=stop if side is TradeSide.LONG else 1.105,
        margin_required=margin,
        pending=pending_order,
        submitted_at=NOW,
    )


def guard(limits: PortfolioRiskLimits | None = None) -> RiskGuard:
    converter = StaticCurrencyConverter({("EUR", "USD"): 1.10})
    return RiskGuard(
        analyzer=PortfolioAnalyzer(converter),
        limits=limits,
        policy_name="portfolio",
        policy_version="2b",
    )


def assert_rejected(decision_codes: tuple[RiskLimitCode, ...], code: RiskLimitCode) -> None:
    assert code in decision_codes


def test_guard_approves_trade_within_all_limits() -> None:
    decision = guard().evaluate(snapshot(), proposal())
    assert decision.approved
    assert decision.status is RiskDecisionStatus.APPROVED
    assert decision.violations == ()
    assert not decision.kill_switch_engaged
    assert decision.metrics_after.open_position_count == 1
    assert dict(decision.audit_fields)["policy_version"] == "2b"


def test_decision_id_is_deterministic() -> None:
    first = guard().evaluate(snapshot(), proposal())
    second = guard().evaluate(snapshot(), proposal())
    assert first.decision_id == second.decision_id
    assert len(first.decision_id) == 24


def test_manual_and_existing_automatic_kill_switches_reject() -> None:
    manual = guard().evaluate(snapshot(manual_kill=True), proposal())
    automatic = guard().evaluate(snapshot(automatic_kill=True), proposal())
    assert_rejected(manual.rejection_codes, RiskLimitCode.MANUAL_KILL_SWITCH)
    assert_rejected(automatic.rejection_codes, RiskLimitCode.AUTOMATIC_KILL_SWITCH)
    assert manual.kill_switch_engaged
    assert automatic.kill_switch_engaged


def test_open_position_limit_rejects_next_market_trade() -> None:
    limits = PortfolioRiskLimits(max_open_positions=1)
    decision = guard(limits).evaluate(snapshot(positions=(position("p1"),)), proposal())
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_OPEN_POSITIONS)


def test_pending_order_limit_rejects_next_pending_order() -> None:
    limits = PortfolioRiskLimits(max_pending_orders=1)
    decision = guard(limits).evaluate(
        snapshot(pending_orders=(pending("o1"),)),
        proposal(pending_order=True),
    )
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_PENDING_ORDERS)


def test_symbol_position_limit_counts_positions_and_pending_orders() -> None:
    limits = PortfolioRiskLimits(max_positions_per_symbol=2)
    decision = guard(limits).evaluate(
        snapshot(
            positions=(position("p1"),),
            pending_orders=(pending("o1"),),
        ),
        proposal(),
    )
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_POSITIONS_PER_SYMBOL)


def test_total_open_risk_limit_rejects() -> None:
    limits = PortfolioRiskLimits(max_total_risk_fraction=0.005)
    decision = guard(limits).evaluate(snapshot(), proposal(volume=2.0, margin=2_200))
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_TOTAL_RISK)


def test_gross_notional_limit_rejects() -> None:
    limits = PortfolioRiskLimits(
        max_gross_notional_multiple=1.0,
        max_net_notional_multiple=100,
        max_currency_concentration_multiple=100,
    )
    decision = guard(limits).evaluate(snapshot(), proposal(volume=0.2, margin=220))
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_GROSS_NOTIONAL)


def test_net_notional_limit_rejects() -> None:
    limits = PortfolioRiskLimits(
        max_gross_notional_multiple=100,
        max_net_notional_multiple=1.0,
        max_currency_concentration_multiple=100,
    )
    decision = guard(limits).evaluate(snapshot(), proposal(volume=0.2, margin=220))
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_NET_NOTIONAL)


def test_offsetting_trade_can_pass_net_limit_but_still_counts_gross() -> None:
    limits = PortfolioRiskLimits(
        max_gross_notional_multiple=3.0,
        max_net_notional_multiple=0.2,
        max_currency_concentration_multiple=100,
    )
    current = position("long", volume=0.1)
    decision = guard(limits).evaluate(
        snapshot(positions=(current,)),
        proposal(side=TradeSide.SHORT, volume=0.1),
    )
    assert RiskLimitCode.MAX_NET_NOTIONAL not in decision.rejection_codes
    assert decision.metrics_after.net_notional_amount == pytest.approx(0)


def test_margin_utilization_limit_rejects() -> None:
    limits = PortfolioRiskLimits(max_margin_utilization=0.05)
    decision = guard(limits).evaluate(snapshot(), proposal(margin=600))
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_MARGIN_UTILIZATION)


def test_currency_concentration_limit_rejects() -> None:
    limits = PortfolioRiskLimits(
        max_gross_notional_multiple=100,
        max_net_notional_multiple=100,
        max_currency_concentration_multiple=0.5,
    )
    decision = guard(limits).evaluate(snapshot(), proposal(volume=0.1))
    assert_rejected(decision.rejection_codes, RiskLimitCode.MAX_CURRENCY_CONCENTRATION)


def test_protective_stop_is_required_by_default() -> None:
    decision = guard().evaluate(snapshot(), proposal(stop=None))
    assert_rejected(decision.rejection_codes, RiskLimitCode.PROTECTIVE_STOP_REQUIRED)


def test_protective_stop_requirement_can_be_disabled() -> None:
    limits = PortfolioRiskLimits(require_protective_stop=False)
    decision = guard(limits).evaluate(snapshot(), proposal(stop=None))
    assert RiskLimitCode.PROTECTIVE_STOP_REQUIRED not in decision.rejection_codes


def test_pending_order_exposure_can_be_excluded() -> None:
    limits = PortfolioRiskLimits(
        include_pending_order_exposure=False,
        max_total_risk_fraction=0.001,
        max_gross_notional_multiple=0.1,
        max_net_notional_multiple=0.1,
        max_currency_concentration_multiple=0.1,
    )
    decision = guard(limits).evaluate(
        snapshot(),
        proposal(volume=10, margin=9_000, pending_order=True),
    )
    assert decision.approved
    assert decision.metrics_after.pending_order_count == 1
    assert decision.metrics_after.pending_risk_amount == 0
    assert decision.metrics_after.gross_notional_amount == 0


def test_daily_realized_loss_triggers_automatic_kill_switch() -> None:
    limits = PortfolioRiskLimits(max_daily_realized_loss_fraction=0.03)
    decision = guard(limits).evaluate(snapshot(realized=-300), proposal())
    assert_rejected(decision.rejection_codes, RiskLimitCode.DAILY_REALIZED_LOSS)
    assert decision.automatic_kill_switch_triggered
    assert decision.kill_switch_engaged


def test_daily_total_loss_triggers_automatic_kill_switch() -> None:
    limits = PortfolioRiskLimits(max_daily_total_loss_fraction=0.04)
    decision = guard(limits).evaluate(snapshot(equity=9_600), proposal())
    assert_rejected(decision.rejection_codes, RiskLimitCode.DAILY_TOTAL_LOSS)
    assert decision.automatic_kill_switch_triggered


def test_intraday_drawdown_triggers_automatic_kill_switch() -> None:
    limits = PortfolioRiskLimits(max_intraday_drawdown_fraction=0.05)
    decision = guard(limits).evaluate(snapshot(equity=9_500, peak=10_000), proposal())
    assert_rejected(decision.rejection_codes, RiskLimitCode.INTRADAY_DRAWDOWN)
    assert decision.automatic_kill_switch_triggered


def test_equity_stop_triggers_automatic_kill_switch() -> None:
    limits = PortfolioRiskLimits(min_equity_fraction_of_balance=0.75)
    decision = guard(limits).evaluate(
        snapshot(balance=10_000, equity=7_500, day_start=7_500, peak=7_500),
        proposal(),
    )
    assert_rejected(decision.rejection_codes, RiskLimitCode.EQUITY_STOP)
    assert decision.automatic_kill_switch_triggered


def test_consecutive_loss_cooldown_blocks_before_expiry() -> None:
    limits = PortfolioRiskLimits(
        max_consecutive_losses=3,
        consecutive_loss_cooldown=timedelta(minutes=30),
    )
    decision = guard(limits).evaluate(
        snapshot(
            consecutive_losses=3,
            last_loss_time=NOW - timedelta(minutes=10),
        ),
        proposal(),
    )
    assert_rejected(decision.rejection_codes, RiskLimitCode.CONSECUTIVE_LOSS_COOLDOWN)


def test_consecutive_loss_cooldown_allows_after_expiry() -> None:
    limits = PortfolioRiskLimits(
        max_consecutive_losses=3,
        consecutive_loss_cooldown=timedelta(minutes=30),
    )
    decision = guard(limits).evaluate(
        snapshot(
            consecutive_losses=3,
            last_loss_time=NOW - timedelta(minutes=31),
        ),
        proposal(),
    )
    assert RiskLimitCode.CONSECUTIVE_LOSS_COOLDOWN not in decision.rejection_codes


def test_missing_last_loss_time_is_conservatively_blocked() -> None:
    limits = PortfolioRiskLimits(max_consecutive_losses=3)
    decision = guard(limits).evaluate(
        snapshot(consecutive_losses=3),
        proposal(),
    )
    assert_rejected(decision.rejection_codes, RiskLimitCode.CONSECUTIVE_LOSS_COOLDOWN)


def test_naive_evaluation_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        guard().evaluate(
            snapshot(),
            proposal(),
            evaluated_at=datetime(2026, 8, 2, 12),
        )


def test_multiple_limit_breaches_are_reported_together() -> None:
    limits = PortfolioRiskLimits(
        max_open_positions=1,
        max_positions_per_symbol=1,
        max_margin_utilization=0.01,
        max_gross_notional_multiple=0.5,
        max_net_notional_multiple=0.5,
        max_currency_concentration_multiple=0.5,
    )
    decision = guard(limits).evaluate(
        snapshot(positions=(position("p1"),)),
        proposal(volume=1, margin=1_100),
    )
    assert not decision.approved
    assert len(decision.violations) >= 4
    assert len(decision.rejection_codes) == len(set(decision.rejection_codes))
