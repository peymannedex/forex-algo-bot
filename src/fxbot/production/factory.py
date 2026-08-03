"""Factory for wiring the Phase 5 execution stack from production settings."""

from __future__ import annotations

from dataclasses import dataclass

from fxbot.execution.adapters import MT5BrokerAdapter, MT5ExecutionConfig
from fxbot.execution.broker import FillSink, OrderSink, RiskAuthorizer
from fxbot.execution.connection import (
    MT5ConnectionManager,
    MT5ExecutionConnectionConfig,
)
from fxbot.execution.reconciliation import LiveMT5Reconciler
from fxbot.execution.router import ExecutionRouter
from fxbot.execution.runtime import ExecutionRuntime
from fxbot.execution.safety import ExecutionControl
from fxbot.production.config import ProductionSettings


@dataclass(frozen=True, slots=True)
class ProductionComponents:
    connection: MT5ConnectionManager
    broker: MT5BrokerAdapter
    control: ExecutionControl
    router: ExecutionRouter
    runtime: ExecutionRuntime
    reconciler: LiveMT5Reconciler


def build_mt5_components(
    settings: ProductionSettings,
    *,
    risk_authorizer: RiskAuthorizer | None = None,
    fill_sinks: tuple[FillSink, ...] = (),
    order_sinks: tuple[OrderSink, ...] = (),
) -> ProductionComponents:
    """Build the MT5 execution stack while preserving the profile's dry-run gate."""

    password = (
        settings.mt5_password.get_secret_value()
        if settings.mt5_password is not None
        else None
    )
    connection = MT5ConnectionManager(
        config=MT5ExecutionConnectionConfig(
            terminal_path=settings.mt5_terminal_path,
            login=settings.mt5_login,
            password=password,
            server=settings.mt5_server,
        )
    )
    broker = MT5BrokerAdapter(
        connection,
        config=MT5ExecutionConfig(
            magic_number=settings.mt5_magic_number,
            deviation_points=settings.mt5_deviation_points,
            dry_run=settings.broker_dry_run,
        ),
    )
    control = ExecutionControl.armed()
    router = ExecutionRouter(
        broker,
        risk_authorizer=risk_authorizer,
        control=control,
    )
    runtime = ExecutionRuntime(
        broker,
        fill_sinks=fill_sinks,
        order_sinks=order_sinks,
    )
    reconciler = LiveMT5Reconciler(broker)
    return ProductionComponents(
        connection=connection,
        broker=broker,
        control=control,
        router=router,
        runtime=runtime,
        reconciler=reconciler,
    )
