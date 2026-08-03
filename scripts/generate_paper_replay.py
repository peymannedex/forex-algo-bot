"""Generate deterministic multi-timeframe EURUSD bars for paper acceptance tests."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/paper/paper_replay.csv"),
    )
    parser.add_argument("--m5-bars", type=int, default=360)
    return parser


def row(
    timestamp: datetime,
    timeframe: str,
    open_price: float,
    close_price: float,
    spread: float,
) -> dict[str, object]:
    high = max(open_price, close_price) + 0.00015
    low = min(open_price, close_price) - 0.00015
    return {
        "timestamp": timestamp.isoformat(),
        "symbol": "EURUSD",
        "timeframe": timeframe,
        "bid_open": f"{open_price:.5f}",
        "bid_high": f"{high:.5f}",
        "bid_low": f"{low:.5f}",
        "bid_close": f"{close_price:.5f}",
        "ask_open": f"{open_price + spread:.5f}",
        "ask_high": f"{high + spread:.5f}",
        "ask_low": f"{low + spread:.5f}",
        "ask_close": f"{close_price + spread:.5f}",
        "tick_volume": 100,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.m5_bars < 180:
        raise SystemExit("--m5-bars must be at least 180")
    output: list[dict[str, object]] = []
    start = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    price = 1.09000
    m5_closes: list[float] = []
    for index in range(args.m5_bars):
        timestamp = start + timedelta(minutes=5 * index)
        phase = (index // 90) % 4
        drift = 0.00006 if phase in {0, 3} else -0.00005
        pulse = 0.00003 if index % 7 in {0, 1, 2} else -0.00001
        close = max(price + drift + pulse, 0.5)
        output.append(row(timestamp, "M5", price, close, 0.00008))
        m5_closes.append(close)
        price = close

        if (index + 1) % 3 == 0:
            block_start = index - 2
            open_price = m5_closes[block_start - 1] if block_start > 0 else 1.09000
            close_price = m5_closes[index]
            output.append(
                row(
                    start + timedelta(minutes=5 * block_start),
                    "M15",
                    open_price,
                    close_price,
                    0.00008,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
