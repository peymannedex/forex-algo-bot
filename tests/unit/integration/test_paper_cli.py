import json

import pytest

from fxbot.integration.cli import build_parser, main

from .test_paper_replay import write_csv


def write_env(path, tmp_path, profile="paper") -> None:
    path.write_text(
        "\n".join(
            [
                f"FXBOT_PROFILE={profile}",
                'FXBOT_SYMBOLS=["EURUSD"]',
                f"FXBOT_STATE_DIRECTORY={tmp_path / 'state'}",
                f"FXBOT_LOG_DIRECTORY={tmp_path / 'log'}",
                "FXBOT_PAPER_FIXED_QUANTITY=0.01",
                "FXBOT_PAPER_WARMUP_BARS=2",
                "FXBOT_PAPER_REQUIRED_TIMEFRAMES=M5,M15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parser_has_expected_default_env_file() -> None:
    args = build_parser().parse_args([])

    assert args.env_file.name == ".env"
    assert not args.reset_state


def test_cli_runs_hold_replay_and_prints_json(tmp_path, capsys) -> None:
    env_file = tmp_path / ".env"
    replay = tmp_path / "replay.csv"
    write_env(env_file, tmp_path)
    write_csv(replay)

    result = main(
        [
            "--env-file",
            str(env_file),
            "--replay-csv",
            str(replay),
            "--reset-state",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["profile"] == "paper"
    assert payload["summary"]["cycles"] == 1
    assert payload["ready"] is True


def test_cli_rejects_demo_profile(tmp_path) -> None:
    env_file = tmp_path / ".env"
    replay = tmp_path / "replay.csv"
    write_env(env_file, tmp_path, profile="demo")
    write_csv(replay)

    with pytest.raises(SystemExit, match="FXBOT_PROFILE=paper"):
        main(["--env-file", str(env_file), "--replay-csv", str(replay)])


def test_cli_smoke_strategy_exercises_orders(tmp_path, capsys) -> None:
    from subprocess import run
    from sys import executable

    env_file = tmp_path / ".env"
    replay = tmp_path / "generated.csv"
    write_env(env_file, tmp_path)
    generation = run(
        [
            executable,
            "scripts/generate_paper_replay.py",
            "--output",
            str(replay),
            "--m5-bars",
            "180",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generation.returncode == 0

    result = main(
        [
            "--env-file",
            str(env_file),
            "--replay-csv",
            str(replay),
            "--strategy",
            "smoke",
            "--reset-state",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["summary"]["accepted_orders"] > 0
    assert payload["summary"]["fills"] > 0
