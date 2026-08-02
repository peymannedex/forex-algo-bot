"""Equal-level liquidity pools, invalidation, and stop-run sweep detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import fsum, isfinite

from fxbot.domain.enums import Timeframe
from fxbot.strategy.context import MarketSeries
from fxbot.strategy.market_structure import StructureDirection, SwingKind, SwingPoint


class LiquiditySide(StrEnum):
    """Resting liquidity expected beyond clustered extrema."""

    BUY_SIDE = "buy_side"
    SELL_SIDE = "sell_side"


def _positive(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class LiquidityConfig:
    """Clustering and sweep-confirmation thresholds expressed in ATR units."""

    equal_level_tolerance_atr: float = 0.10
    minimum_touches: int = 2
    sweep_buffer_atr: float = 0.02
    close_reentry_tolerance_atr: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "equal_level_tolerance_atr",
            _non_negative(
                self.equal_level_tolerance_atr,
                "equal_level_tolerance_atr",
            ),
        )
        if self.minimum_touches < 2:
            raise ValueError("minimum_touches must be at least 2")
        object.__setattr__(
            self,
            "sweep_buffer_atr",
            _non_negative(self.sweep_buffer_atr, "sweep_buffer_atr"),
        )
        object.__setattr__(
            self,
            "close_reentry_tolerance_atr",
            _non_negative(
                self.close_reentry_tolerance_atr,
                "close_reentry_tolerance_atr",
            ),
        )


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    """Cluster of equal highs or lows with a durable lifecycle."""

    symbol: str
    timeframe: Timeframe
    side: LiquiditySide
    level: float
    swing_indices: tuple[int, ...]
    formed_at: datetime
    tolerance: float
    invalidated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", LiquiditySide(self.side))
        object.__setattr__(self, "level", _positive(self.level, "level"))
        object.__setattr__(self, "tolerance", _non_negative(self.tolerance, "tolerance"))
        if len(self.swing_indices) < 2:
            raise ValueError("Liquidity pools require at least two swing indices")
        if tuple(sorted(set(self.swing_indices))) != self.swing_indices:
            raise ValueError("swing_indices must be sorted and unique")
        object.__setattr__(self, "formed_at", _utc(self.formed_at, "formed_at"))
        if self.invalidated_at is not None:
            invalidated = _utc(self.invalidated_at, "invalidated_at")
            if invalidated < self.formed_at:
                raise ValueError("invalidated_at cannot precede formed_at")
            object.__setattr__(self, "invalidated_at", invalidated)

    @property
    def active(self) -> bool:
        return self.invalidated_at is None


@dataclass(frozen=True, slots=True)
class LiquiditySweep:
    """Intrabar penetration of a pool followed by a close back through the level."""

    symbol: str
    timeframe: Timeframe
    index: int
    swept_side: LiquiditySide
    direction: StructureDirection
    pool_level: float
    extreme_price: float
    close_price: float
    penetration: float
    event_time: datetime
    pool: LiquidityPool

    def __post_init__(self) -> None:
        object.__setattr__(self, "swept_side", LiquiditySide(self.swept_side))
        direction = StructureDirection(self.direction)
        if direction is StructureDirection.NEUTRAL:
            raise ValueError("Liquidity sweeps must be directional")
        expected = (
            StructureDirection.BEARISH
            if self.swept_side is LiquiditySide.BUY_SIDE
            else StructureDirection.BULLISH
        )
        if direction is not expected:
            raise ValueError("Sweep direction does not match swept liquidity side")
        object.__setattr__(self, "direction", direction)
        if self.index < 0:
            raise ValueError("index cannot be negative")
        object.__setattr__(self, "pool_level", _positive(self.pool_level, "pool_level"))
        object.__setattr__(self, "extreme_price", _positive(self.extreme_price, "extreme_price"))
        object.__setattr__(self, "close_price", _positive(self.close_price, "close_price"))
        object.__setattr__(self, "penetration", _positive(self.penetration, "penetration"))
        object.__setattr__(self, "event_time", _utc(self.event_time, "event_time"))


def detect_liquidity_pools(
    series: MarketSeries,
    swings: tuple[SwingPoint, ...],
    *,
    atr: float,
    config: LiquidityConfig | None = None,
) -> tuple[LiquidityPool, ...]:
    """Cluster confirmed equal highs/lows and track close-through invalidation."""

    settings = config or LiquidityConfig()
    volatility = _positive(atr, "atr")
    tolerance = volatility * settings.equal_level_tolerance_atr
    result: list[LiquidityPool] = []

    for kind, side in (
        (SwingKind.HIGH, LiquiditySide.BUY_SIDE),
        (SwingKind.LOW, LiquiditySide.SELL_SIDE),
    ):
        selected = [swing for swing in swings if swing.kind is kind]
        clusters: list[list[SwingPoint]] = []
        for swing in selected:
            placed = False
            for cluster in clusters:
                mean = fsum(item.price for item in cluster) / len(cluster)
                if abs(swing.price - mean) <= tolerance:
                    cluster.append(swing)
                    placed = True
                    break
            if not placed:
                clusters.append([swing])

        for cluster in clusters:
            if len(cluster) < settings.minimum_touches:
                continue
            level = fsum(item.price for item in cluster) / len(cluster)
            formed_at = max(item.confirmed_at for item in cluster)
            invalidated_at: datetime | None = None
            for bar in series.bars:
                if bar.close_time <= formed_at:
                    continue
                if side is LiquiditySide.BUY_SIDE and bar.mid.close > level + tolerance:
                    invalidated_at = bar.close_time
                    break
                if side is LiquiditySide.SELL_SIDE and bar.mid.close < level - tolerance:
                    invalidated_at = bar.close_time
                    break
            result.append(
                LiquidityPool(
                    symbol=series.symbol,
                    timeframe=series.timeframe,
                    side=side,
                    level=level,
                    swing_indices=tuple(item.index for item in cluster),
                    formed_at=formed_at,
                    tolerance=tolerance,
                    invalidated_at=invalidated_at,
                )
            )
    return tuple(sorted(result, key=lambda item: (item.formed_at, item.side.value)))


def detect_liquidity_sweeps(
    series: MarketSeries,
    pools: tuple[LiquidityPool, ...],
    *,
    atr: float,
    config: LiquidityConfig | None = None,
) -> tuple[LiquiditySweep, ...]:
    """Return wick penetrations that close back inside their liquidity level."""

    settings = config or LiquidityConfig()
    volatility = _positive(atr, "atr")
    sweep_buffer = volatility * settings.sweep_buffer_atr
    reentry = volatility * settings.close_reentry_tolerance_atr
    sweeps: list[LiquiditySweep] = []

    for pool in pools:
        for index, bar in enumerate(series.bars):
            if bar.close_time <= pool.formed_at:
                continue
            if pool.invalidated_at is not None and bar.close_time >= pool.invalidated_at:
                break
            candle = bar.mid
            if pool.side is LiquiditySide.BUY_SIDE:
                penetration = candle.high - pool.level
                confirmed = penetration > sweep_buffer and candle.close <= pool.level + reentry
                direction = StructureDirection.BEARISH
                extreme = candle.high
            else:
                penetration = pool.level - candle.low
                confirmed = penetration > sweep_buffer and candle.close >= pool.level - reentry
                direction = StructureDirection.BULLISH
                extreme = candle.low
            if confirmed:
                sweeps.append(
                    LiquiditySweep(
                        symbol=series.symbol,
                        timeframe=series.timeframe,
                        index=index,
                        swept_side=pool.side,
                        direction=direction,
                        pool_level=pool.level,
                        extreme_price=extreme,
                        close_price=candle.close,
                        penetration=penetration,
                        event_time=bar.close_time,
                        pool=pool,
                    )
                )
                break
    return tuple(sorted(sweeps, key=lambda item: item.event_time))


def nearest_liquidity_target(
    pools: tuple[LiquidityPool, ...],
    *,
    direction: StructureDirection,
    price: float,
) -> LiquidityPool | None:
    """Return the nearest active opposing pool beyond current price."""

    current = _positive(price, "price")
    parsed = StructureDirection(direction)
    if parsed is StructureDirection.BULLISH:
        candidates = [
            pool
            for pool in pools
            if pool.active and pool.side is LiquiditySide.BUY_SIDE and pool.level > current
        ]
        return min(candidates, key=lambda item: item.level) if candidates else None
    if parsed is StructureDirection.BEARISH:
        candidates = [
            pool
            for pool in pools
            if pool.active and pool.side is LiquiditySide.SELL_SIDE and pool.level < current
        ]
        return max(candidates, key=lambda item: item.level) if candidates else None
    return None
