import pytest

from fxbot.integration.live_config import PaperLiveFeedSettings


def test_live_settings_defaults_are_safe() -> None:
    settings = PaperLiveFeedSettings()

    assert settings.source == "mt5"
    assert settings.max_consecutive_errors == 5
    assert settings.stop_filename == "STOP_PAPER_SOAK"
    assert settings.mt5_server_utc_offset_minutes == 0
    assert settings.max_future_skew_seconds == 5.0


def test_stop_filename_rejects_paths() -> None:
    with pytest.raises(ValueError, match="plain file name"):
        PaperLiveFeedSettings(stop_filename="../stop")


def test_server_offset_is_bounded() -> None:
    with pytest.raises(ValueError, match="within"):
        PaperLiveFeedSettings(mt5_server_utc_offset_minutes=1_441)


def test_future_skew_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PaperLiveFeedSettings(max_future_skew_seconds=-0.1)
