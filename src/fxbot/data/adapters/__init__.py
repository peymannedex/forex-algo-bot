"""Market-data adapter implementations."""

from fxbot.data.adapters.base import (
    AdapterDiagnostics,
    HistoricalMarketDataAdapter,
    LiveMarketDataAdapter,
    MarketDataAdapterError,
)
from fxbot.data.adapters.csv_adapter import CSVMarketDataAdapter
from fxbot.data.adapters.mt5_adapter import (
    MT5AdapterError,
    MT5ConnectionConfig,
    MT5MarketDataAdapter,
)
from fxbot.data.adapters.parquet_adapter import ParquetMarketDataAdapter
from fxbot.data.adapters.queue_adapter import AsyncQueueLiveDataAdapter

__all__ = [
    "AdapterDiagnostics",
    "AsyncQueueLiveDataAdapter",
    "CSVMarketDataAdapter",
    "HistoricalMarketDataAdapter",
    "LiveMarketDataAdapter",
    "MT5AdapterError",
    "MT5ConnectionConfig",
    "MT5MarketDataAdapter",
    "MarketDataAdapterError",
    "ParquetMarketDataAdapter",
]
