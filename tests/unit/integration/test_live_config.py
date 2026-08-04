
import pytest

from fxbot.integration.live_config import PaperLiveFeedSettings


def test_live_settings_defaults_are_safe() -> None:
    settings = PaperLiveFeedSettings()

    assert settings.source == "mt5"
    assert settings.max_consecutive_errors == 5
    assert settings.stop_filename == "STOP_PAPER_SOAK"


def test_stop_filename_rejects_paths() -> None:
    with pytest.raises(ValueError, match="plain file name"):
        PaperLiveFeedSettings(stop_filename="../stop")
