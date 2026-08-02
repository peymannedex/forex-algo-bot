"""Look-ahead-safe single and multi-timeframe market context."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from fxbot.domain.enums import Timeframe
from fxbot.domain.models import Bar
from fxbot.strategy.models import StrategyConfig


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class ContextIssueCode(StrEnum):
    """Machine-readable reasons a strategy context is not executable."""

    MISSING_TIMEFRAME = "missing_timeframe"
    INSUFFICIENT_WARMUP = "insufficient_warmup"
    STALE_PRIMARY_DATA = "stale_primary_data"
    INCOMPLETE_BAR = "incomplete_bar"
    FUTURE_DATA = "future_data"


@dataclass(frozen=True, slots=True)
class ContextIssue:
    code: ContextIssueCode
    message: str
    timeframe: Timeframe | None = None
    observed: float | None = None
    required: float | None = None


@dataclass(frozen=True, slots=True)
class MarketSeries:
    """Chronologically ordered bars for one symbol and timeframe."""

    symbol: str
    timeframe: Timeframe
    bars: tuple[Bar, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        timeframe = Timeframe.parse(self.timeframe)
        if timeframe is Timeframe.TICK:
            raise ValueError("MarketSeries timeframe cannot be tick")
        object.__setattr__(self, "timeframe", timeframe)

        previous_open: datetime | None = None
        for bar in self.bars:
            if bar.symbol != self.symbol:
                raise ValueError(f"Bar symbol {bar.symbol} does not match {self.symbol}")
            if bar.timeframe is not timeframe:
                raise ValueError(
                    f"Bar timeframe {bar.timeframe.value} does not match {timeframe.value}"
                )
            if previous_open is not None and bar.open_time <= previous_open:
                raise ValueError("Bars must be strictly increasing with no duplicates")
            previous_open = bar.open_time

    @property
    def latest(self) -> Bar:
        if not self.bars:
            raise LookupError("MarketSeries is empty")
        return self.bars[-1]

    @property
    def closes(self) -> tuple[float, ...]:
        return tuple(bar.mid.close for bar in self.bars)

    @property
    def highs(self) -> tuple[float, ...]:
        return tuple(bar.mid.high for bar in self.bars)

    @property
    def lows(self) -> tuple[float, ...]:
        return tuple(bar.mid.low for bar in self.bars)

    @property
    def spreads(self) -> tuple[float, ...]:
        return tuple(bar.spread_close for bar in self.bars)

    def window(self, size: int) -> MarketSeries:
        if size < 1:
            raise ValueError("size must be positive")
        return MarketSeries(self.symbol, self.timeframe, self.bars[-size:])

    def available_at(self, as_of: datetime) -> MarketSeries:
        """Return only bars whose information was available at ``as_of``."""

        cutoff = _utc(as_of, "as_of")
        available = tuple(
            bar
            for bar in self.bars
            if (bar.close_time <= cutoff if bar.complete else bar.open_time <= cutoff)
        )
        return MarketSeries(self.symbol, self.timeframe, available)


@dataclass(frozen=True, slots=True)
class MultiTimeframeContext:
    """Consistent point-in-time market view used by strategies."""

    symbol: str
    as_of: datetime
    primary_timeframe: Timeframe
    series: tuple[MarketSeries, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        primary = Timeframe.parse(self.primary_timeframe)
        if primary is Timeframe.TICK:
            raise ValueError("primary_timeframe cannot be tick")
        object.__setattr__(self, "primary_timeframe", primary)

        seen: set[Timeframe] = set()
        for item in self.series:
            if item.symbol != self.symbol:
                raise ValueError(f"Series symbol {item.symbol} does not match {self.symbol}")
            if item.timeframe in seen:
                raise ValueError(f"Duplicate series for timeframe {item.timeframe.value}")
            seen.add(item.timeframe)
        if primary not in seen:
            raise ValueError("Primary timeframe series is required")

    @property
    def primary(self) -> MarketSeries:
        return self.get(self.primary_timeframe)

    @property
    def timeframes(self) -> tuple[Timeframe, ...]:
        return tuple(item.timeframe for item in self.series)

    def get(self, timeframe: Timeframe) -> MarketSeries:
        parsed = Timeframe.parse(timeframe)
        for item in self.series:
            if item.timeframe is parsed:
                return item
        raise KeyError(f"Context does not contain timeframe {parsed.value}")

    def validate(self, config: StrategyConfig) -> tuple[ContextIssue, ...]:
        """Return all readiness issues without mutating or hiding data."""

        if config.primary_timeframe is not self.primary_timeframe:
            raise ValueError("StrategyConfig primary timeframe does not match context")

        issues: list[ContextIssue] = []
        available = set(self.timeframes)
        for timeframe in config.required_timeframes:
            if timeframe not in available:
                issues.append(
                    ContextIssue(
                        code=ContextIssueCode.MISSING_TIMEFRAME,
                        message=f"Missing required timeframe {timeframe.value}",
                        timeframe=timeframe,
                    )
                )
                continue
            market_series = self.get(timeframe)
            if len(market_series.bars) < config.warmup_bars:
                issues.append(
                    ContextIssue(
                        code=ContextIssueCode.INSUFFICIENT_WARMUP,
                        message=(
                            f"{timeframe.value} has {len(market_series.bars)} bars; "
                            f"{config.warmup_bars} required"
                        ),
                        timeframe=timeframe,
                        observed=float(len(market_series.bars)),
                        required=float(config.warmup_bars),
                    )
                )
            has_future_data = False
            has_incomplete_bar = False
            for bar in market_series.bars:
                availability_time = bar.close_time if bar.complete else bar.open_time
                if availability_time > self.as_of:
                    has_future_data = True
                if not config.allow_incomplete_bars and not bar.complete:
                    has_incomplete_bar = True
            if has_future_data:
                issues.append(
                    ContextIssue(
                        code=ContextIssueCode.FUTURE_DATA,
                        message=f"{timeframe.value} contains data unavailable at context time",
                        timeframe=timeframe,
                    )
                )
            if has_incomplete_bar:
                issues.append(
                    ContextIssue(
                        code=ContextIssueCode.INCOMPLETE_BAR,
                        message=f"{timeframe.value} contains an incomplete bar",
                        timeframe=timeframe,
                    )
                )

        primary = self.primary
        if primary.bars:
            latest = primary.latest
            latest_available = latest.close_time if latest.complete else latest.open_time
            age = self.as_of - latest_available
            if age > config.max_data_age:
                issues.append(
                    ContextIssue(
                        code=ContextIssueCode.STALE_PRIMARY_DATA,
                        message=f"Primary data age {age} exceeds {config.max_data_age}",
                        timeframe=self.primary_timeframe,
                        observed=age.total_seconds(),
                        required=config.max_data_age.total_seconds(),
                    )
                )
        return tuple(issues)

    def ready(self, config: StrategyConfig) -> bool:
        return not self.validate(config)


class MarketContextBuilder:
    """Group raw bars into a point-in-time, look-ahead-safe context."""

    def build(
        self,
        *,
        symbol: str,
        as_of: datetime,
        primary_timeframe: Timeframe,
        bars: Iterable[Bar],
    ) -> MultiTimeframeContext:
        normalized_symbol = _symbol(symbol)
        cutoff = _utc(as_of, "as_of")
        grouped: defaultdict[Timeframe, list[Bar]] = defaultdict(list)
        for bar in bars:
            if bar.symbol != normalized_symbol:
                raise ValueError(f"Bar symbol {bar.symbol} does not match {normalized_symbol}")
            availability_time = bar.close_time if bar.complete else bar.open_time
            if availability_time <= cutoff:
                grouped[bar.timeframe].append(bar)

        series = tuple(
            MarketSeries(normalized_symbol, timeframe, tuple(sorted(items, key=lambda x: x.open_time)))
            for timeframe, items in sorted(grouped.items(), key=lambda pair: pair[0].value)
        )
        return MultiTimeframeContext(
            symbol=normalized_symbol,
            as_of=cutoff,
            primary_timeframe=primary_timeframe,
            series=series,
        )


def timeframe_age_tolerance(timeframe: Timeframe, extra: timedelta) -> timedelta:
    """Utility for callers that want timeframe-aware stale-data thresholds."""

    parsed = Timeframe.parse(timeframe)
    seconds = parsed.seconds
    if seconds is None:
        raise ValueError("Tick timeframe does not have a bar-age tolerance")
    if extra < timedelta(0):
        raise ValueError("extra cannot be negative")
    return timedelta(seconds=seconds) + extra
