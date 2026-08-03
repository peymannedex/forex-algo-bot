"""Production configuration, safeguards, health, and operational supervision."""

from fxbot.production.alerts import (
    AlertSeverity,
    AlertSink,
    FanoutAlertSink,
    InMemoryAlertSink,
    LoggingAlertSink,
    OperationalAlert,
    alert,
)
from fxbot.production.checkpoint import (
    SupervisorCheckpoint,
    SupervisorCheckpointStore,
)
from fxbot.production.config import DeploymentProfile, ProductionSettings
from fxbot.production.emergency import (
    EmergencyBroker,
    EmergencyController,
    EmergencyResult,
)
from fxbot.production.factory import (
    ProductionComponents,
    build_mt5_components,
)
from fxbot.production.health import (
    ComponentHealth,
    ComponentState,
    HealthRegistry,
    HealthSnapshot,
)
from fxbot.production.journal import (
    ExecutionJournalEntry,
    FileExecutionJournal,
    JournalState,
    RecoverableFillSink,
)
from fxbot.production.logging import JsonFormatter, configure_json_logging
from fxbot.production.protections import (
    AccountRiskSnapshot,
    LossGuard,
    MarketHoursGuard,
    MarketWindow,
    ProtectionDecision,
    QuoteGuard,
    QuoteProtectionConfig,
)
from fxbot.production.readiness import (
    ReadinessCheck,
    ReadinessReport,
    StartupReadinessGate,
)
from fxbot.production.supervisor import (
    ProductionSupervisor,
    SupervisorCycleResult,
)

__all__ = [
    "AccountRiskSnapshot",
    "AlertSeverity",
    "AlertSink",
    "ComponentHealth",
    "ComponentState",
    "DeploymentProfile",
    "EmergencyBroker",
    "EmergencyController",
    "EmergencyResult",
    "ExecutionJournalEntry",
    "FanoutAlertSink",
    "FileExecutionJournal",
    "HealthRegistry",
    "HealthSnapshot",
    "InMemoryAlertSink",
    "JournalState",
    "JsonFormatter",
    "LoggingAlertSink",
    "LossGuard",
    "MarketHoursGuard",
    "MarketWindow",
    "OperationalAlert",
    "ProductionComponents",
    "ProductionSettings",
    "ProductionSupervisor",
    "ProtectionDecision",
    "QuoteGuard",
    "QuoteProtectionConfig",
    "ReadinessCheck",
    "ReadinessReport",
    "RecoverableFillSink",
    "StartupReadinessGate",
    "SupervisorCheckpoint",
    "SupervisorCheckpointStore",
    "SupervisorCycleResult",
    "alert",
    "build_mt5_components",
    "configure_json_logging",
]
