from pathlib import Path

import pytest
from pydantic import ValidationError

from fxbot.production.config import DeploymentProfile, ProductionSettings


def test_defaults_are_safe() -> None:
    settings = ProductionSettings()

    assert settings.profile is DeploymentProfile.PAPER
    assert settings.broker_dry_run
    assert settings.symbols == ("EURUSD",)


def test_symbols_normalize_and_deduplicate() -> None:
    settings = ProductionSettings(symbols=("eurusd", " EURUSD ", "gbpusd"))

    assert settings.symbols == ("EURUSD", "GBPUSD")


def test_live_profile_requires_explicit_confirmation() -> None:
    with pytest.raises(ValidationError, match="live_trading_enabled"):
        ProductionSettings(profile="live")


def test_live_profile_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="MT5 login"):
        ProductionSettings(
            profile="live",
            live_trading_enabled=True,
            live_confirmation="I_UNDERSTAND_LIVE_TRADING",
        )


def test_live_profile_can_be_enabled_deliberately() -> None:
    settings = ProductionSettings(
        profile="live",
        live_trading_enabled=True,
        live_confirmation="I_UNDERSTAND_LIVE_TRADING",
        mt5_login=123,
        mt5_password="secret",
        mt5_server="Demo",
    )

    assert not settings.broker_dry_run


def test_redacted_never_exposes_secrets() -> None:
    settings = ProductionSettings(
        mt5_password="secret",
        live_confirmation="phrase",
        state_directory=Path("state"),
    )

    output = settings.redacted()

    assert output["mt5_password"] == "***"
    assert output["live_confirmation"] == "***"
    assert "secret" not in repr(output)



def test_demo_submission_requires_explicit_enablement() -> None:
    safe = ProductionSettings(profile="demo")
    enabled = ProductionSettings(
        profile="demo",
        demo_order_submission_enabled=True,
    )

    assert safe.broker_dry_run
    assert not enabled.broker_dry_run
