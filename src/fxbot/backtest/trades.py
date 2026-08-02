"""Closed-trade statistics, streaks, holding times, and excursion analytics."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from statistics import median

from fxbot.backtest.broker import ClosedTrade


def _non_negative(value: float, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class TradeExcursion:
    """Maximum adverse and favorable account-currency movement for one trade."""

    trade_id: str
    maximum_adverse_excursion: float
    maximum_favorable_excursion: float

    def __post_init__(self) -> None:
        trade_id = self.trade_id.strip()
        if not trade_id:
            raise ValueError("trade_id cannot be empty")
        object.__setattr__(self, "trade_id", trade_id)
        object.__setattr__(
            self,
            "maximum_adverse_excursion",
            _non_negative(self.maximum_adverse_excursion, "maximum_adverse_excursion"),
        )
        object.__setattr__(
            self,
            "maximum_favorable_excursion",
            _non_negative(self.maximum_favorable_excursion, "maximum_favorable_excursion"),
        )


@dataclass(frozen=True, slots=True)
class TradeStatistics:
    """Auditable aggregate statistics for a chronologically ordered trade ledger."""

    trade_count: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    average_trade: float
    average_win: float
    average_loss: float
    median_trade: float
    expectancy: float
    payoff_ratio: float
    profit_factor: float
    maximum_consecutive_wins: int
    maximum_consecutive_losses: int
    average_holding_seconds: float
    median_holding_seconds: float
    total_commission: float
    average_mae: float | None = None
    average_mfe: float | None = None


def analyze_trades(
    trades: tuple[ClosedTrade, ...],
    *,
    excursions: tuple[TradeExcursion, ...] = (),
) -> TradeStatistics:
    """Return deterministic trade statistics without mutating the source ledger."""

    ordered = tuple(sorted(trades, key=lambda item: (item.closed_at, item.trade_id)))
    pnl = tuple(float(item.net_pnl) for item in ordered)
    winners = tuple(value for value in pnl if value > 0.0)
    losers = tuple(value for value in pnl if value < 0.0)
    breakeven = sum(value == 0.0 for value in pnl)
    count = len(pnl)
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    net_profit = sum(pnl)
    average_trade = net_profit / count if count else 0.0
    average_win = gross_profit / len(winners) if winners else 0.0
    average_loss = sum(losers) / len(losers) if losers else 0.0
    payoff_ratio = _ratio(average_win, abs(average_loss))
    profit_factor = _ratio(gross_profit, gross_loss)
    holding = tuple((item.closed_at - item.opened_at).total_seconds() for item in ordered)
    if any(value < 0.0 for value in holding):
        raise ValueError("Trade closed_at cannot predate opened_at")

    excursion_map = {item.trade_id: item for item in excursions}
    if len(excursion_map) != len(excursions):
        raise ValueError("Excursion trade IDs must be unique")
    unknown = set(excursion_map) - {item.trade_id for item in ordered}
    if unknown:
        raise ValueError(f"Excursions reference unknown trades: {sorted(unknown)}")
    matched = tuple(excursion_map[item.trade_id] for item in ordered if item.trade_id in excursion_map)

    return TradeStatistics(
        trade_count=count,
        winning_trades=len(winners),
        losing_trades=len(losers),
        breakeven_trades=breakeven,
        win_rate=len(winners) / count if count else 0.0,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        average_trade=average_trade,
        average_win=average_win,
        average_loss=average_loss,
        median_trade=float(median(pnl)) if pnl else 0.0,
        expectancy=average_trade,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        maximum_consecutive_wins=_maximum_streak(pnl, positive=True),
        maximum_consecutive_losses=_maximum_streak(pnl, positive=False),
        average_holding_seconds=sum(holding) / len(holding) if holding else 0.0,
        median_holding_seconds=float(median(holding)) if holding else 0.0,
        total_commission=sum(float(item.commission) for item in ordered),
        average_mae=(
            sum(item.maximum_adverse_excursion for item in matched) / len(matched)
            if matched
            else None
        ),
        average_mfe=(
            sum(item.maximum_favorable_excursion for item in matched) / len(matched)
            if matched
            else None
        ),
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return inf if numerator > 0.0 else 0.0


def _maximum_streak(values: tuple[float, ...], *, positive: bool) -> int:
    longest = 0
    current = 0
    for value in values:
        matched = value > 0.0 if positive else value < 0.0
        current = current + 1 if matched else 0
        longest = max(longest, current)
    return longest
