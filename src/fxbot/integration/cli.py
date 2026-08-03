"""Command-line entry point for deterministic paper replay acceptance testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fxbot.integration.config import PaperIntegrationSettings
from fxbot.integration.factory import (
    build_default_strategy,
    build_paper_components,
    build_smoke_strategy,
)
from fxbot.integration.replay import iter_paper_frames, load_replay_bars, run_replay
from fxbot.production.config import DeploymentProfile, ProductionSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper integration replay")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("C:/forex-algo-bot/config/.env"),
    )
    parser.add_argument("--replay-csv", type=Path, default=None)
    parser.add_argument(
        "--strategy",
        choices=("trend", "smoke"),
        default="trend",
        help="Use smoke only for deterministic execution-path acceptance",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete paper state and fill journal before replay",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    production = ProductionSettings(_env_file=args.env_file)  # type: ignore[call-arg]
    paper = PaperIntegrationSettings(_env_file=args.env_file)  # type: ignore[call-arg]
    if production.profile is not DeploymentProfile.PAPER:
        raise SystemExit("Paper integration requires FXBOT_PROFILE=paper")

    replay_path = args.replay_csv or paper.replay_csv
    state_path = production.state_directory / paper.state_filename
    journal_path = production.state_directory / paper.journal_filename
    if args.reset_state:
        state_path.unlink(missing_ok=True)
        journal_path.unlink(missing_ok=True)

    strategy = (
        build_smoke_strategy(paper)
        if args.strategy == "smoke"
        else build_default_strategy(paper)
    )
    components = build_paper_components(production, paper, strategy=strategy)
    bars = load_replay_bars(replay_path)
    frames = iter_paper_frames(
        bars,
        primary_timeframe=paper.parsed_primary_timeframe,
    )
    if not frames:
        raise SystemExit("Replay contains no primary-timeframe frames")
    summary, _ = run_replay(components.runtime, frames)
    payload = {
        "ready": components.health.snapshot().ready,
        "profile": production.profile.value,
        "replay_csv": str(replay_path),
        "summary": summary.to_dict(),
        "health": components.health.snapshot().to_dict(),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
