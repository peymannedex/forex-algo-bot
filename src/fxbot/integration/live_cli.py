"""CLI for sustained live-market-data paper operation."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path

from fxbot.domain.models import LiveSubscription
from fxbot.integration.config import PaperIntegrationSettings
from fxbot.integration.factory import (
    build_default_strategy,
    build_paper_components,
    build_smoke_strategy,
)
from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.integration.live_feed import MT5ReadOnlyMarketSource
from fxbot.integration.soak import (
    LivePaperFrameAssembler,
    PaperLiveSoakRunner,
    SoakEvidenceWriter,
)
from fxbot.production.config import DeploymentProfile, ProductionSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only live-feed paper service"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("C:/forex-algo-bot/config/.env"),
    )
    parser.add_argument(
        "--strategy",
        choices=("trend", "smoke"),
        default="trend",
    )
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete paper checkpoint and journal before starting",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    production = ProductionSettings(  # type: ignore[call-arg]
        _env_file=args.env_file
    )
    paper = PaperIntegrationSettings(  # type: ignore[call-arg]
        _env_file=args.env_file
    )
    live = PaperLiveFeedSettings(  # type: ignore[call-arg]
        _env_file=args.env_file
    )

    if production.profile is not DeploymentProfile.PAPER:
        raise SystemExit("Live-feed soak requires FXBOT_PROFILE=paper")
    if production.live_trading_enabled:
        raise SystemExit("FXBOT_LIVE_TRADING_ENABLED must remain false")
    if production.demo_order_submission_enabled:
        raise SystemExit(
            "FXBOT_DEMO_ORDER_SUBMISSION_ENABLED must remain false"
        )
    if live.source != "mt5":
        raise SystemExit(f"Unsupported paper live source: {live.source}")

    state_path = production.state_directory / paper.state_filename
    journal_path = production.state_directory / paper.journal_filename
    stop_file = production.state_directory / live.stop_filename

    if args.reset_state:
        state_path.unlink(missing_ok=True)
        journal_path.unlink(missing_ok=True)
    stop_file.unlink(missing_ok=True)

    strategy = (
        build_smoke_strategy(paper)
        if args.strategy == "smoke"
        else build_default_strategy(paper)
    )
    components = build_paper_components(
        production,
        paper,
        strategy=strategy,
    )
    source = MT5ReadOnlyMarketSource(production, live)
    subscription = LiveSubscription(
        symbols=frozenset(production.symbols),
        timeframes=frozenset(paper.parsed_required_timeframes),
    )
    assembler = LivePaperFrameAssembler(
        primary_timeframe=paper.parsed_primary_timeframe,
        required_timeframes=paper.parsed_required_timeframes,
        history_limit=live.history_bars_per_timeframe,
    )
    evidence = SoakEvidenceWriter(
        live.evidence_directory,
        health=components.health,
    )
    runner = PaperLiveSoakRunner(
        source=source,
        runtime=components.runtime,
        health=components.health,
        subscription=subscription,
        assembler=assembler,
        evidence=evidence,
        settings=live,
        stop_file=stop_file,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for signal_name in ("SIGINT", "SIGTERM"):
        resolved = getattr(signal, signal_name, None)
        if resolved is None:
            continue
        try:
            loop.add_signal_handler(resolved, request_stop)
        except (NotImplementedError, RuntimeError):
            signal.signal(resolved, lambda *_: request_stop())

    startup = {
        "profile": production.profile.value,
        "source": live.source,
        "symbols": list(production.symbols),
        "timeframes": [
            item.value for item in paper.parsed_required_timeframes
        ],
        "strategy": args.strategy,
        "paper_broker_only": True,
        "mt5_order_submission": False,
        "mt5_server_utc_offset_minutes": (
            live.mt5_server_utc_offset_minutes
        ),
        "max_future_skew_seconds": live.max_future_skew_seconds,
        "evidence_directory": str(live.evidence_directory),
        "stop_file": str(stop_file),
    }
    print(json.dumps(startup, sort_keys=True), flush=True)

    summary = await runner.run(
        stop_event,
        max_cycles=args.max_cycles,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(summary.to_dict(), sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
