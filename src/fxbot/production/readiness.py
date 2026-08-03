"""Startup readiness validation for paper, demo, and live profiles."""

from __future__ import annotations

from dataclasses import dataclass

from fxbot.execution.connection import MT5ConnectionSnapshot
from fxbot.production.config import DeploymentProfile, ProductionSettings


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[ReadinessCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


class StartupReadinessGate:
    def __init__(self, settings: ProductionSettings) -> None:
        self.settings = settings

    def evaluate(
        self,
        connection: MT5ConnectionSnapshot,
        *,
        broker_dry_run: bool,
    ) -> ReadinessReport:
        checks = [
            ReadinessCheck(
                "terminal_connected",
                connection.connected,
                "MT5 terminal connected"
                if connection.connected
                else "MT5 terminal disconnected",
            ),
            ReadinessCheck(
                "account_available",
                connection.account_login is not None,
                "MT5 account available"
                if connection.account_login is not None
                else "MT5 account unavailable",
            ),
        ]
        requires_trade = self.settings.profile in {
            DeploymentProfile.DEMO,
            DeploymentProfile.LIVE,
        }
        checks.append(
            ReadinessCheck(
                "trade_allowed",
                connection.trade_allowed or not requires_trade,
                "algorithmic trading allowed"
                if connection.trade_allowed
                else "algorithmic trading not allowed",
            )
        )
        if self.settings.profile is DeploymentProfile.LIVE:
            checks.append(
                ReadinessCheck(
                    "live_submission_enabled",
                    not broker_dry_run,
                    "live broker submission enabled"
                    if not broker_dry_run
                    else "broker remains in dry-run mode",
                )
            )
        elif self.settings.profile is DeploymentProfile.PAPER:
            checks.append(
                ReadinessCheck(
                    "paper_dry_run",
                    broker_dry_run,
                    "paper profile cannot submit broker orders"
                    if broker_dry_run
                    else "paper profile attempted live broker submission",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "demo_submission_mode",
                    True,
                    "demo order submission enabled"
                    if not broker_dry_run
                    else "demo broker remains in validation-only mode",
                )
            )
        return ReadinessReport(tuple(checks))
