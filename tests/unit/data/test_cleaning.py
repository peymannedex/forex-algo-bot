from datetime import UTC, datetime, timedelta

from fxbot.data.cleaning import DataCleaningConfig, MarketDataCleaner
from fxbot.domain.models import SymbolSpec, Tick


def test_cleaner_sorts_deduplicates_and_rejects_wide_spread() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    normal = Tick("EURUSD", t0, 1.1000, 1.1001)
    duplicate = Tick("EURUSD", t0, 1.1000, 1.1001)
    wide = Tick("EURUSD", t0 + timedelta(seconds=1), 1.1000, 1.1010)
    spec = SymbolSpec("EURUSD", "EUR", "USD", 5, 0.00001, 0.0001)

    cleaner = MarketDataCleaner(
        DataCleaningConfig(max_spread_pips=3.0),
        {"EURUSD": spec},
    )
    result = cleaner.clean_ticks([wide, duplicate, normal])

    assert result.records == (normal,)
    assert result.report.duplicates_removed == 1
    assert result.report.spread_rejections == 1
    assert result.report.reordered is True
