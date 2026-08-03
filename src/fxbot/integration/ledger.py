"""Netted paper portfolio ledger and conservative mark-to-market accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from fxbot.execution.models import ExecutionFill, Quote
from fxbot.integration.models import PaperAccountView, PaperPosition
from fxbot.integration.state import PaperRuntimeState
from fxbot.production.protections import AccountRiskSnapshot
from fxbot.risk.models import AccountSnapshot


@dataclass(slots=True)
class _MutablePosition:
    signed_quantity: float
    average_price: float
    realized_pnl: float = 0.0


class PaperPortfolioLedger:
    """Track paper fills, positions, realized PnL, and marked account equity."""

    def __init__(
        self,
        *,
        initial_balance: float = 100_000.0,
        currency: str = "USD",
        leverage: float = 100.0,
        contract_sizes: dict[str, float] | None = None,
    ) -> None:
        balance = float(initial_balance)
        leverage_value = float(leverage)
        if not isfinite(balance) or balance <= 0.0:
            raise ValueError("initial_balance must be positive and finite")
        if not isfinite(leverage_value) or leverage_value < 1.0:
            raise ValueError("leverage must be finite and at least one")
        normalized_currency = currency.strip().upper()
        if not normalized_currency:
            raise ValueError("currency cannot be empty")
        self.currency = normalized_currency
        self.leverage = leverage_value
        self.initial_balance = balance
        self.balance = balance
        self.day_start_equity = balance
        self.peak_equity = balance
        self.realized_pnl = 0.0
        self._positions: dict[str, _MutablePosition] = {}
        self._quotes: dict[str, Quote] = {}
        self._contract_sizes = {
            symbol.strip().upper(): self._positive(value, "contract_size")
            for symbol, value in (contract_sizes or {}).items()
        }
        self._updated_at = datetime.now(UTC)

    @staticmethod
    def _positive(value: float, field_name: str) -> float:
        number = float(value)
        if not isfinite(number) or number <= 0.0:
            raise ValueError(f"{field_name} must be positive and finite")
        return number

    def contract_size(self, symbol: str) -> float:
        return self._contract_sizes.get(symbol.strip().upper(), 100_000.0)

    def on_quote(self, quote: Quote) -> None:
        current = self._quotes.get(quote.symbol)
        if current is not None and quote.timestamp < current.timestamp:
            raise ValueError("quotes must be chronological per symbol")
        self._quotes[quote.symbol] = quote
        self._updated_at = max(self._updated_at, quote.timestamp)
        self.peak_equity = max(self.peak_equity, self.equity)

    def on_fill(self, fill: ExecutionFill) -> None:
        symbol = fill.symbol
        delta = fill.quantity * fill.side.sign
        contract_size = self.contract_size(symbol)
        current = self._positions.get(symbol)
        realized = -fill.commission

        if current is None:
            self._positions[symbol] = _MutablePosition(delta, fill.price, realized)
        elif current.signed_quantity * delta > 0.0:
            total = abs(current.signed_quantity) + abs(delta)
            average = (
                current.average_price * abs(current.signed_quantity)
                + fill.price * abs(delta)
            ) / total
            current.signed_quantity += delta
            current.average_price = average
            current.realized_pnl += realized
        else:
            closing_quantity = min(abs(current.signed_quantity), abs(delta))
            direction = 1.0 if current.signed_quantity > 0.0 else -1.0
            realized += (
                (fill.price - current.average_price)
                * closing_quantity
                * contract_size
                * direction
            )
            remaining = current.signed_quantity + delta
            current.realized_pnl += realized
            if abs(remaining) <= 1e-12:
                del self._positions[symbol]
            elif current.signed_quantity * remaining > 0.0:
                current.signed_quantity = remaining
            else:
                self._positions[symbol] = _MutablePosition(
                    remaining,
                    fill.price,
                    current.realized_pnl,
                )

        self.balance += realized
        self.realized_pnl += realized
        self._updated_at = max(self._updated_at, fill.executed_at)
        if self.balance <= 0.0:
            raise RuntimeError("paper account balance became non-positive")
        self.peak_equity = max(self.peak_equity, self.equity)

    def quote(self, symbol: str) -> Quote | None:
        return self._quotes.get(symbol.strip().upper())

    def signed_position(self, symbol: str) -> float:
        position = self._positions.get(symbol.strip().upper())
        return position.signed_quantity if position is not None else 0.0

    def expected_positions(self) -> dict[str, float]:
        return {
            symbol: position.signed_quantity
            for symbol, position in sorted(self._positions.items())
        }

    @property
    def gross_quantity(self) -> float:
        return sum(abs(position.signed_quantity) for position in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        total = 0.0
        for symbol, position in self._positions.items():
            quote = self._quotes.get(symbol)
            if quote is None:
                continue
            mark = quote.bid if position.signed_quantity > 0.0 else quote.ask
            direction = 1.0 if position.signed_quantity > 0.0 else -1.0
            total += (
                (mark - position.average_price)
                * abs(position.signed_quantity)
                * self.contract_size(symbol)
                * direction
            )
        return total

    @property
    def equity(self) -> float:
        return self.balance + self.unrealized_pnl

    @property
    def margin_used(self) -> float:
        notional = 0.0
        for symbol, position in self._positions.items():
            quote = self._quotes.get(symbol)
            mark = quote.mid if quote is not None else position.average_price
            notional += abs(position.signed_quantity) * self.contract_size(symbol) * mark
        return notional / self.leverage

    def account_snapshot(self) -> AccountSnapshot:
        equity = self.equity
        margin = self.margin_used
        return AccountSnapshot(
            currency=self.currency,
            balance=self.balance,
            equity=equity,
            free_margin=max(equity - margin, 0.0),
            margin_used=margin,
            leverage=self.leverage,
        )

    def account_risk_snapshot(self, *, checked_at: datetime) -> AccountRiskSnapshot:
        return AccountRiskSnapshot(
            equity=self.equity,
            daily_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity,
            checked_at=checked_at,
        )

    def view(self) -> PaperAccountView:
        positions = tuple(
            PaperPosition(
                symbol=symbol,
                signed_quantity=position.signed_quantity,
                average_price=position.average_price,
                realized_pnl=position.realized_pnl,
            )
            for symbol, position in sorted(self._positions.items())
        )
        return PaperAccountView(
            currency=self.currency,
            balance=self.balance,
            equity=self.equity,
            day_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            positions=positions,
            updated_at=self._updated_at,
        )

    def state(self, *, cycle: int, last_frame_at: datetime | None) -> PaperRuntimeState:
        return PaperRuntimeState(
            cycle=cycle,
            last_frame_at=last_frame_at,
            balance=self.balance,
            day_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity,
            realized_pnl=self.realized_pnl,
            positions=self.view().positions,
        )

    def restore(self, state: PaperRuntimeState) -> None:
        self.balance = state.balance
        self.day_start_equity = state.day_start_equity
        self.peak_equity = state.peak_equity
        self.realized_pnl = state.realized_pnl
        self._positions = {
            position.symbol: _MutablePosition(
                position.signed_quantity,
                position.average_price,
                position.realized_pnl,
            )
            for position in state.positions
        }
        if state.last_frame_at is not None:
            self._updated_at = state.last_frame_at
