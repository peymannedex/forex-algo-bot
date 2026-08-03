"""Atomic paper-runtime state persistence for restart recovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fxbot.integration.models import PaperPosition


def _float_value(value: object, field_name: str) -> float:
    """Convert a serialized scalar to float with a narrow type contract."""

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{field_name} must be a numeric scalar")
    return float(value)


@dataclass(frozen=True, slots=True)
class PaperRuntimeState:
    cycle: int = 0
    last_frame_at: datetime | None = None
    balance: float = 100_000.0
    day_start_equity: float = 100_000.0
    peak_equity: float = 100_000.0
    realized_pnl: float = 0.0
    positions: tuple[PaperPosition, ...] = ()

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError("cycle cannot be negative")
        if self.last_frame_at is not None:
            if self.last_frame_at.tzinfo is None or self.last_frame_at.utcoffset() is None:
                raise ValueError("last_frame_at must be timezone-aware")
            object.__setattr__(self, "last_frame_at", self.last_frame_at.astimezone(UTC))
        for name in ("balance", "day_start_equity", "peak_equity"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "cycle": self.cycle,
            "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
            "balance": self.balance,
            "day_start_equity": self.day_start_equity,
            "peak_equity": self.peak_equity,
            "realized_pnl": self.realized_pnl,
            "positions": [
                {
                    "symbol": position.symbol,
                    "signed_quantity": position.signed_quantity,
                    "average_price": position.average_price,
                    "realized_pnl": position.realized_pnl,
                }
                for position in self.positions
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> PaperRuntimeState:
        last_frame = raw.get("last_frame_at")
        raw_positions = raw.get("positions", [])
        if not isinstance(raw_positions, list):
            raise ValueError("positions must be a list")
        positions: list[PaperPosition] = []
        for raw_position in raw_positions:
            if not isinstance(raw_position, dict):
                raise ValueError("position state must be an object")
            item = cast(dict[str, object], raw_position)
            positions.append(
                PaperPosition(
                    symbol=str(item["symbol"]),
                    signed_quantity=_float_value(
                        item["signed_quantity"],
                        "signed_quantity",
                    ),
                    average_price=_float_value(
                        item["average_price"],
                        "average_price",
                    ),
                    realized_pnl=_float_value(
                        item.get("realized_pnl", 0.0),
                        "realized_pnl",
                    ),
                )
            )
        return cls(
            cycle=int(str(raw.get("cycle", 0))),
            last_frame_at=datetime.fromisoformat(str(last_frame)) if last_frame else None,
            balance=float(str(raw.get("balance", 100_000.0))),
            day_start_equity=float(str(raw.get("day_start_equity", 100_000.0))),
            peak_equity=float(str(raw.get("peak_equity", 100_000.0))),
            realized_pnl=float(str(raw.get("realized_pnl", 0.0))),
            positions=tuple(positions),
        )


class PaperRuntimeStateStore:
    """Persist paper runtime state with an atomic file replacement."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> PaperRuntimeState | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("paper runtime state root must be an object")
        return PaperRuntimeState.from_dict(cast(dict[str, object], raw))

    def save(self, state: PaperRuntimeState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
