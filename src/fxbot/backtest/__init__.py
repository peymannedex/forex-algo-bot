"""Deterministic event-driven historical simulation."""

from fxbot.backtest.broker import (
    BrokerSnapshot,
    ClosedTrade,
    NetPosition,
    SimulatedBroker,
)
from fxbot.backtest.clock import HistoricalClock, HistoricalClockError
from fxbot.backtest.config import (
    BacktestConfig,
    CommissionConfig,
    ExecutionConfig,
    InstrumentConfig,
    SlippageConfig,
    SwapConfig,
)
from fxbot.backtest.engine import (
    AllowAllRiskGate,
    BacktestEngine,
    BacktestStrategy,
    FillObserver,
    RiskGate,
    RiskGateResult,
    StrategyDecisionAdapter,
)
from fxbot.backtest.events import (
    AuditEvent,
    BacktestEvent,
    EventKind,
    MarketEvent,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    SimulatedFill,
    TimeInForce,
    market_record_time,
)
from fxbot.backtest.results import BacktestResult, EquityPoint, equity_point

__all__ = [
    "AllowAllRiskGate",
    "AuditEvent",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestEvent",
    "BacktestResult",
    "BacktestStrategy",
    "BrokerSnapshot",
    "ClosedTrade",
    "CommissionConfig",
    "EquityPoint",
    "EventKind",
    "ExecutionConfig",
    "FillObserver",
    "HistoricalClock",
    "HistoricalClockError",
    "InstrumentConfig",
    "MarketEvent",
    "NetPosition",
    "OrderRequest",
    "OrderSide",
    "OrderState",
    "OrderStatus",
    "OrderType",
    "RiskGate",
    "RiskGateResult",
    "SimulatedBroker",
    "SimulatedFill",
    "SlippageConfig",
    "StrategyDecisionAdapter",
    "SwapConfig",
    "TimeInForce",
    "equity_point",
    "market_record_time",
]
