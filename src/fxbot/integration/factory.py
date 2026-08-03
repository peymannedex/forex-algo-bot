"""Factory wiring the merged strategy, risk, execution, and production layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta

from fxbot.domain.models import SymbolSpec
from fxbot.execution.paper import PaperBroker, PaperBrokerConfig
from fxbot.execution.router import ExecutionRouter
from fxbot.execution.runtime import ExecutionRuntime
from fxbot.execution.safety import ExecutionControl
from fxbot.integration.config import PaperIntegrationSettings
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.planner import (
    DecisionOrderPlanner,
    FixedQuantityPolicy,
    PositionSizerQuantityPolicy,
    QuantityPolicy,
)
from fxbot.integration.risk import GuardedPaperRiskAuthorizer, PaperExposureLimits
from fxbot.integration.runtime import PaperIntegrationRuntime
from fxbot.integration.smoke import AcceptanceSmokeStrategy
from fxbot.integration.state import PaperRuntimeStateStore
from fxbot.production.config import ProductionSettings
from fxbot.production.health import HealthRegistry
from fxbot.production.journal import FileExecutionJournal, RecoverableFillSink
from fxbot.production.protections import (
    LossGuard,
    MarketHoursGuard,
    MarketWindow,
    QuoteGuard,
    QuoteProtectionConfig,
)
from fxbot.risk.models import BrokerVolumeConstraints, InstrumentRiskSpec
from fxbot.risk.position_sizing import IdentityCurrencyConverter, PositionSizer
from fxbot.strategy.base import Strategy, StrategyRuntime
from fxbot.strategy.models import StrategyConfig
from fxbot.strategy.trend_following import TrendFollowingConfig, TrendFollowingStrategy


@dataclass(frozen=True, slots=True)
class PaperIntegrationComponents:
    broker: PaperBroker
    ledger: PaperPortfolioLedger
    control: ExecutionControl
    risk_authorizer: GuardedPaperRiskAuthorizer
    router: ExecutionRouter
    execution_runtime: ExecutionRuntime
    strategy_runtime: StrategyRuntime
    planner: DecisionOrderPlanner
    health: HealthRegistry
    state_store: PaperRuntimeStateStore
    runtime: PaperIntegrationRuntime


def default_instrument(symbol: str) -> InstrumentRiskSpec:
    """Return a conservative spot-FX instrument specification for paper acceptance."""

    normalized = symbol.strip().upper()
    if len(normalized) != 6 or not normalized.isalpha():
        raise ValueError(f"Cannot infer FX currencies from symbol {symbol!r}")
    base = normalized[:3]
    quote = normalized[3:]
    jpy = quote == "JPY"
    point_size = 0.001 if jpy else 0.00001
    pip_size = 0.01 if jpy else 0.0001
    digits = 3 if jpy else 5
    return InstrumentRiskSpec(
        symbol=SymbolSpec(
            symbol=normalized,
            base_currency=base,
            quote_currency=quote,
            digits=digits,
            point_size=point_size,
            pip_size=pip_size,
            contract_size=100_000.0,
        ),
        volume=BrokerVolumeConstraints(0.01, 100.0, 0.01),
        tick_size=point_size,
    )


def build_smoke_strategy(settings: PaperIntegrationSettings) -> AcceptanceSmokeStrategy:
    """Build the deterministic acceptance-only strategy."""

    return AcceptanceSmokeStrategy(
        StrategyConfig(
            strategy_id="paper-acceptance-smoke",
            primary_timeframe=settings.parsed_primary_timeframe,
            warmup_bars=1,
            min_confidence=0.0,
        )
    )


def build_default_strategy(settings: PaperIntegrationSettings) -> TrendFollowingStrategy:
    strategy = StrategyConfig(
        strategy_id="paper-trend-following",
        primary_timeframe=settings.parsed_primary_timeframe,
        required_timeframes=settings.parsed_required_timeframes,
        warmup_bars=settings.warmup_bars,
        min_confidence=settings.min_confidence,
        duplicate_suppression_window=timedelta(
            seconds=settings.duplicate_suppression_seconds
        ),
    )
    return TrendFollowingStrategy(TrendFollowingConfig(strategy=strategy))


def build_paper_components(
    production: ProductionSettings,
    paper: PaperIntegrationSettings,
    *,
    strategy: Strategy | None = None,
) -> PaperIntegrationComponents:
    """Construct the complete restart-safe paper integration runtime."""

    if production.profile.value != "paper":
        raise ValueError("paper integration requires FXBOT_PROFILE=paper")

    instruments = {
        symbol: default_instrument(symbol)
        for symbol in production.symbols
    }
    contract_sizes = {
        symbol: instrument.symbol.contract_size
        for symbol, instrument in instruments.items()
    }
    ledger = PaperPortfolioLedger(
        initial_balance=paper.initial_balance,
        currency=paper.account_currency,
        leverage=paper.leverage,
        contract_sizes=contract_sizes,
    )
    broker = PaperBroker(
        PaperBrokerConfig(
            max_fill_quantity_per_quote=paper.max_fill_quantity_per_quote,
            commission_per_unit=paper.commission_per_unit,
            slippage=paper.slippage,
        )
    )
    control = ExecutionControl.armed()
    quote_guard = QuoteGuard(
        QuoteProtectionConfig(
            max_age_seconds=production.max_quote_age_seconds,
            max_spread_bps=production.max_spread_bps,
        )
    )
    loss_guard = LossGuard(
        control,
        max_daily_loss=production.max_daily_loss,
        max_drawdown=production.max_drawdown,
    )
    market_guard = MarketHoursGuard(
        (
            MarketWindow(
                weekdays=frozenset({0, 1, 2, 3, 4}),
                start=time(0, 0),
                end=time(23, 59, 59),
            ),
        )
    )
    risk_authorizer = GuardedPaperRiskAuthorizer(
        ledger=ledger,
        quote_guard=quote_guard,
        loss_guard=loss_guard,
        market_hours_guard=market_guard,
        limits=PaperExposureLimits(
            max_abs_position_per_symbol=paper.max_abs_position_per_symbol,
            max_gross_quantity=paper.max_gross_quantity,
        ),
    )
    router = ExecutionRouter(
        broker,
        risk_authorizer=risk_authorizer,
        control=control,
    )
    journal = FileExecutionJournal(
        production.state_directory / paper.journal_filename
    )
    recoverable_ledger = RecoverableFillSink(journal, ledger)
    execution_runtime = ExecutionRuntime(
        broker,
        fill_sinks=(recoverable_ledger,),
    )

    quantity_policy: QuantityPolicy
    if paper.fixed_quantity is not None:
        quantity_policy = FixedQuantityPolicy(paper.fixed_quantity)
    else:
        for instrument in instruments.values():
            if instrument.symbol.quote_currency != paper.account_currency:
                raise ValueError(
                    "risk-based paper sizing currently requires instrument quote "
                    "currency to match account_currency; configure fixed_quantity otherwise"
                )
        quantity_policy = PositionSizerQuantityPolicy(
            sizer=PositionSizer(converter=IdentityCurrencyConverter()),
            instruments=instruments,
            risk_fraction=paper.risk_fraction,
        )

    planner = DecisionOrderPlanner(quantity_policy)
    health = HealthRegistry(
        stale_after=timedelta(seconds=production.health_stale_after_seconds)
    )
    state_store = PaperRuntimeStateStore(
        production.state_directory / paper.state_filename
    )
    resolved_strategy = strategy or build_default_strategy(paper)
    strategy_runtime = StrategyRuntime()
    runtime = PaperIntegrationRuntime(
        strategy=resolved_strategy,
        strategy_runtime=strategy_runtime,
        planner=planner,
        broker=broker,
        router=router,
        execution_runtime=execution_runtime,
        ledger=ledger,
        health=health,
        state_store=state_store,
    )
    return PaperIntegrationComponents(
        broker=broker,
        ledger=ledger,
        control=control,
        risk_authorizer=risk_authorizer,
        router=router,
        execution_runtime=execution_runtime,
        strategy_runtime=strategy_runtime,
        planner=planner,
        health=health,
        state_store=state_store,
        runtime=runtime,
    )
