from pathlib import Path

import pytest

from fxbot.domain.enums import Timeframe
from fxbot.integration.config import PaperIntegrationSettings
from fxbot.integration.factory import (
    build_paper_components,
    default_instrument,
)
from fxbot.production.config import DeploymentProfile, ProductionSettings


def production(tmp_path: Path, profile: DeploymentProfile = DeploymentProfile.PAPER):
    return ProductionSettings(
        profile=profile,
        state_directory=tmp_path / "state",
        log_directory=tmp_path / "log",
        symbols=("EURUSD",),
    )


def test_timeframe_parsing_adds_primary_first() -> None:
    settings = PaperIntegrationSettings(
        primary_timeframe="M5",
        required_timeframes="M15,M5,H1",
    )

    assert settings.parsed_required_timeframes == (
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1,
    )


def test_tick_primary_is_rejected() -> None:
    settings = PaperIntegrationSettings(primary_timeframe="tick")

    with pytest.raises(ValueError, match="cannot be tick"):
        _ = settings.parsed_primary_timeframe


def test_default_instrument_parses_eurusd() -> None:
    instrument = default_instrument("eurusd")

    assert instrument.symbol.base_currency == "EUR"
    assert instrument.symbol.quote_currency == "USD"
    assert instrument.volume.minimum == 0.01


def test_default_instrument_handles_jpy_digits() -> None:
    instrument = default_instrument("USDJPY")

    assert instrument.symbol.digits == 3
    assert instrument.symbol.pip_size == 0.01


def test_invalid_fx_symbol_rejected() -> None:
    with pytest.raises(ValueError, match="infer"):
        default_instrument("GOLD")


def test_factory_builds_restart_safe_stack(tmp_path) -> None:
    components = build_paper_components(
        production(tmp_path),
        PaperIntegrationSettings(fixed_quantity=0.01),
    )

    assert components.broker.name == "paper"
    assert components.runtime.cycle == 0
    assert components.state_store.path.parent == tmp_path / "state"


def test_factory_rejects_non_paper_profile(tmp_path) -> None:
    demo = production(tmp_path, DeploymentProfile.DEMO)

    with pytest.raises(ValueError, match="FXBOT_PROFILE=paper"):
        build_paper_components(
            demo,
            PaperIntegrationSettings(fixed_quantity=0.01),
        )


def test_risk_sizing_requires_matching_quote_currency(tmp_path) -> None:
    settings = production(tmp_path)
    settings = settings.model_copy(update={"symbols": ("EURJPY",)})

    with pytest.raises(ValueError, match="quote currency"):
        build_paper_components(settings, PaperIntegrationSettings())
