from pydantic import SecretStr

from fxbot.integration.live_config import PaperLiveFeedSettings
from fxbot.integration.live_feed import MT5ReadOnlyMarketSource
from fxbot.production.config import ProductionSettings


def test_mt5_source_is_market_data_only() -> None:
    production = ProductionSettings(
        profile="paper",
        mt5_login=123,
        mt5_password=SecretStr("secret"),
        mt5_server="Demo",
    )
    source = MT5ReadOnlyMarketSource(
        production,
        PaperLiveFeedSettings(),
    )

    assert hasattr(source, "connect")
    assert hasattr(source, "stream")
    assert not hasattr(source, "submit_order")
