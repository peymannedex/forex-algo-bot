"""Paper integration runtime joining market data, strategy, risk, and execution."""

from fxbot.integration.config import PaperIntegrationSettings
from fxbot.integration.factory import (
    PaperIntegrationComponents,
    build_default_strategy,
    build_paper_components,
    build_smoke_strategy,
    default_instrument,
)
from fxbot.integration.ledger import PaperPortfolioLedger
from fxbot.integration.models import (
    PaperAccountView,
    PaperCycleResult,
    PaperFrame,
    PaperOrderOutcome,
    PaperPosition,
)
from fxbot.integration.planner import (
    DecisionOrderPlanner,
    FixedQuantityPolicy,
    PositionSizerQuantityPolicy,
    QuantityPolicy,
)
from fxbot.integration.replay import (
    PaperReplaySummary,
    iter_paper_frames,
    load_replay_bars,
    run_replay,
)
from fxbot.integration.risk import GuardedPaperRiskAuthorizer, PaperExposureLimits
from fxbot.integration.runtime import PaperIntegrationRuntime
from fxbot.integration.smoke import AcceptanceSmokeStrategy
from fxbot.integration.state import PaperRuntimeState, PaperRuntimeStateStore

__all__ = [
    "AcceptanceSmokeStrategy",
    "DecisionOrderPlanner",
    "FixedQuantityPolicy",
    "GuardedPaperRiskAuthorizer",
    "PaperAccountView",
    "PaperCycleResult",
    "PaperExposureLimits",
    "PaperFrame",
    "PaperIntegrationComponents",
    "PaperIntegrationRuntime",
    "PaperIntegrationSettings",
    "PaperOrderOutcome",
    "PaperPortfolioLedger",
    "PaperPosition",
    "PaperReplaySummary",
    "PaperRuntimeState",
    "PaperRuntimeStateStore",
    "PositionSizerQuantityPolicy",
    "QuantityPolicy",
    "build_default_strategy",
    "build_paper_components",
    "build_smoke_strategy",
    "default_instrument",
    "iter_paper_frames",
    "load_replay_bars",
    "run_replay",
]
