from datetime import UTC, datetime

from fxbot.data.adapters.csv_adapter import CSVMarketDataAdapter
from fxbot.data.schemas import TickCSVSchema
from fxbot.domain.enums import DataKind, ParseErrorPolicy
from fxbot.domain.models import HistoricalDataRequest


def test_csv_adapter_streams_and_filters_ticks(tmp_path) -> None:
    path = tmp_path / "ticks.csv"
    path.write_text(
        "timestamp,symbol,bid,ask\n"
        "2026-01-01T00:00:00Z,EURUSD,1.1000,1.1002\n"
        "2026-01-01T00:00:01Z,GBPUSD,1.2500,1.2502\n",
        encoding="utf-8",
    )
    adapter = CSVMarketDataAdapter(
        path,
        tick_schema=TickCSVSchema(
            bid_size=None,
            ask_size=None,
            sequence=None,
        ),
        error_policy=ParseErrorPolicy.RAISE,
    )
    request = HistoricalDataRequest(
        symbol="EURUSD",
        kind=DataKind.TICK,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )

    records = list(adapter.iter_ticks(request))

    assert len(records) == 1
    assert records[0].symbol == "EURUSD"
    assert adapter.diagnostics.rows_read == 2
    assert adapter.diagnostics.records_emitted == 1
