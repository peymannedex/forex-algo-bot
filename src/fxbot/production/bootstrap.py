"""Safe command-line preflight for production configuration and MT5 readiness."""

from __future__ import annotations

import json
import logging

from fxbot.production.config import DeploymentProfile, ProductionSettings
from fxbot.production.factory import build_mt5_components
from fxbot.production.logging import configure_json_logging
from fxbot.production.readiness import StartupReadinessGate


def main() -> int:
    settings = ProductionSettings()
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    settings.log_directory.mkdir(parents=True, exist_ok=True)
    logger = configure_json_logging(
        level=logging.INFO,
        log_file=settings.log_directory / "preflight.jsonl",
    )
    logger.info(
        "Production preflight started",
        extra={"settings": settings.redacted()},
    )

    if settings.profile is DeploymentProfile.PAPER:
        print(
            json.dumps(
                {
                    "ready": True,
                    "profile": settings.profile.value,
                    "message": "Paper profile validated; no live terminal connection attempted.",
                    "settings": settings.redacted(),
                },
                sort_keys=True,
            )
        )
        return 0

    components = build_mt5_components(settings)
    try:
        snapshot = components.connection.ensure_connected()
        report = StartupReadinessGate(settings).evaluate(
            snapshot,
            broker_dry_run=components.broker.config.dry_run,
        )
        print(
            json.dumps(
                {
                    "ready": report.ready,
                    "profile": settings.profile.value,
                    "checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "message": check.message,
                        }
                        for check in report.checks
                    ],
                },
                sort_keys=True,
            )
        )
        return 0 if report.ready else 2
    finally:
        components.connection.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
