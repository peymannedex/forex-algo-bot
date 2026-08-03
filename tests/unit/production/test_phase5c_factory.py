from fxbot.production.config import ProductionSettings
from fxbot.production.factory import build_mt5_components


def test_factory_enforces_dry_run_for_paper() -> None:
    components = build_mt5_components(ProductionSettings())

    assert components.broker.config.dry_run
    assert components.control.state.enabled


def test_factory_disables_dry_run_only_for_confirmed_live() -> None:
    settings = ProductionSettings(
        profile="live",
        live_trading_enabled=True,
        live_confirmation="I_UNDERSTAND_LIVE_TRADING",
        mt5_login=123,
        mt5_password="secret",
        mt5_server="Demo",
    )

    components = build_mt5_components(settings)

    assert not components.broker.config.dry_run
