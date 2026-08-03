"""MetaTrader 5 live broker adapter implementing the Phase 5A protocol."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

from fxbot.execution.broker import (
    OrderNotFoundError,
    PermanentBrokerError,
)
from fxbot.execution.connection import MT5ConnectionManager
from fxbot.execution.models import (
    BrokerOrder,
    ExecutionFill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
)
from fxbot.execution.mt5_mapping import (
    MT5SymbolSpec,
    build_mt5_request,
    client_order_comment,
    order_type_from_mt5,
    side_from_mt5_order_type,
    status_from_mt5_state,
)
from fxbot.execution.mt5_recovery import (
    classification_from_result,
    raise_for_mt5_result,
    record_time,
    unique_by_ticket,
)
from fxbot.execution.reconciliation import MT5PositionSnapshot


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _identifier(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(number)


@dataclass(frozen=True, slots=True)
class MT5ExecutionConfig:
    """Live execution behavior independent of terminal credentials."""

    magic_number: int = 51001
    deviation_points: int = 20
    dry_run: bool = False
    history_lookback: timedelta = timedelta(days=7)
    include_external_orders: bool = False

    def __post_init__(self) -> None:
        if self.magic_number <= 0:
            raise ValueError("magic_number must be positive")
        if self.deviation_points < 0:
            raise ValueError("deviation_points must be non-negative")
        if self.history_lookback <= timedelta(0):
            raise ValueError("history_lookback must be positive")


class MT5BrokerAdapter:
    """Synchronous live MT5 adapter with recovery-safe client identities."""

    def __init__(
        self,
        connection: MT5ConnectionManager,
        *,
        config: MT5ExecutionConfig | None = None,
    ) -> None:
        self.connection = connection
        self.config = config or MT5ExecutionConfig()
        self._orders: dict[str, BrokerOrder] = {}
        self._client_by_broker: dict[str, str] = {}
        self._comment_by_client: dict[str, str] = {}
        self._seen_deals: set[str] = set()
        self._pending_fills: list[ExecutionFill] = []
        self._last_deal_poll = datetime.now(UTC) - self.config.history_lookback
        self._lock = RLock()

    @property
    def name(self) -> str:
        return "mt5"

    @property
    def client(self) -> Any:
        return self.connection.client

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        with self._lock:
            info = self.connection.ensure_symbol(intent.symbol)
            spec = MT5SymbolSpec.from_mt5(intent.symbol, info)
            quote = self._quote(intent.symbol)
            try:
                request = build_mt5_request(
                    self.client,
                    intent,
                    spec,
                    quote,
                    magic_number=self.config.magic_number,
                    deviation_points=self.config.deviation_points,
                )
            except ValueError as exc:
                raise PermanentBrokerError(str(exc)) from exc
            self._comment_by_client[intent.client_order_id] = str(request["comment"])
            self._apply_reduce_only(intent, request)

            check = self.client.order_check(request)
            raise_for_mt5_result(self.client, check, "order_check")
            if self.config.dry_run:
                now = datetime.now(UTC)
                order = BrokerOrder(
                    broker_order_id=f"dry-{intent.client_order_id}",
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    order_type=intent.order_type,
                    status=OrderStatus.ACKNOWLEDGED,
                    requested_quantity=float(request["volume"]),
                    filled_quantity=0.0,
                    average_fill_price=None,
                    submitted_at=now,
                    updated_at=now,
                    metadata=(("dry_run", "true"), ("comment", str(request["comment"]))),
                )
                self._cache_order(order)
                return order

            result = self.client.order_send(request)
            raise_for_mt5_result(self.client, result, "order_send")
            order = self._order_from_submission(intent, request, result, quote)
            self._cache_order(order)
            self._queue_submission_fill(order, result)
            return order

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        with self._lock:
            order = self.get_order(broker_order_id)
            if order.status.terminal:
                return order
            if broker_order_id.startswith("dry-"):
                cancelled = replace(
                    order,
                    status=OrderStatus.CANCELLED,
                    updated_at=datetime.now(UTC),
                )
                self._cache_order(cancelled)
                return cancelled
            try:
                ticket = int(broker_order_id)
            except ValueError as exc:
                raise PermanentBrokerError(f"Invalid MT5 broker order ID: {broker_order_id}") from exc
            request = {
                "action": int(self.client.TRADE_ACTION_REMOVE),
                "order": ticket,
                "magic": self.config.magic_number,
                "comment": "fxb:cancel",
            }
            result = self.client.order_send(request)
            raise_for_mt5_result(self.client, result, "cancel_order")
            cancelled = replace(
                order,
                status=OrderStatus.CANCELLED,
                updated_at=datetime.now(UTC),
            )
            self._cache_order(cancelled)
            return cancelled

    def get_order(self, broker_order_id: str) -> BrokerOrder:
        with self._lock:
            cached = self._orders.get(broker_order_id)
            if cached is not None and (cached.status.terminal or broker_order_id.startswith("dry-")):
                return cached
            try:
                ticket = int(broker_order_id)
            except ValueError:
                if cached is not None:
                    return cached
                raise OrderNotFoundError(f"MT5 order not found: {broker_order_id}") from None

            self.connection.ensure_connected()
            active = self.client.orders_get(ticket=ticket)
            records = tuple(active or ())
            if not records:
                history = self.client.history_orders_get(ticket=ticket)
                records = tuple(history or ())
            if not records:
                if cached is not None:
                    return cached
                raise OrderNotFoundError(f"MT5 order not found: {broker_order_id}")
            order = self._order_from_raw(records[0])
            self._cache_order(order)
            return order

    def find_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        with self._lock:
            for order in self._orders.values():
                if order.client_order_id == client_order_id:
                    return order
            self.connection.ensure_connected()
            expected_comment = client_order_comment(client_order_id)
            records = list(self.client.orders_get() or ())
            end = datetime.now(UTC)
            start = end - self.config.history_lookback
            records.extend(self.client.history_orders_get(start, end) or ())
            for raw in unique_by_ticket(records):
                if not self._belongs_to_adapter(raw):
                    continue
                if str(_field(raw, "comment", "")) != expected_comment:
                    continue
                ticket = _identifier(_field(raw, "ticket"))
                self._client_by_broker[ticket] = client_order_id
                self._comment_by_client[client_order_id] = expected_comment
                order = self._order_from_raw(raw)
                self._cache_order(order)
                return order
            return None

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        self.connection.ensure_connected()
        output: list[BrokerOrder] = []
        for raw in self.client.orders_get() or ():
            if not self._belongs_to_adapter(raw):
                continue
            order = self._order_from_raw(raw)
            self._cache_order(order)
            if order.status.active:
                output.append(order)
        output.sort(key=lambda item: (item.submitted_at, item.broker_order_id))
        return tuple(output)

    def drain_fills(self) -> tuple[ExecutionFill, ...]:
        with self._lock:
            now = datetime.now(UTC)
            recovered = self._fills_from_history(self._last_deal_poll, now, mark_seen=True)
            self._last_deal_poll = now
            output = self._pending_fills + list(recovered)
            self._pending_fills = []
            unique: list[ExecutionFill] = []
            seen: set[str] = set()
            for fill in output:
                if fill.execution_id in seen:
                    continue
                seen.add(fill.execution_id)
                unique.append(fill)
            unique.sort(key=lambda item: (item.executed_at, item.execution_id))
            return tuple(unique)

    def update_quote(self, quote: Quote) -> tuple[ExecutionFill, ...]:
        del quote
        return ()

    def snapshot_open_orders(self) -> tuple[BrokerOrder, ...]:
        return self.list_open_orders()

    def snapshot_positions(self) -> tuple[MT5PositionSnapshot, ...]:
        self.connection.ensure_connected()
        output: list[MT5PositionSnapshot] = []
        buy_type = int(self.client.POSITION_TYPE_BUY)
        for raw in self.client.positions_get() or ():
            if not self._belongs_to_adapter(raw):
                continue
            volume = float(_field(raw, "volume", 0.0))
            sign = 1.0 if int(_field(raw, "type", buy_type)) == buy_type else -1.0
            output.append(
                MT5PositionSnapshot(
                    position_id=_identifier(_field(raw, "ticket")),
                    symbol=str(_field(raw, "symbol")),
                    signed_quantity=sign * volume,
                    average_price=float(_field(raw, "price_open")),
                    profit=float(_field(raw, "profit", 0.0)),
                    updated_at=record_time(raw),
                )
            )
        output.sort(key=lambda item: (item.symbol, item.position_id))
        return tuple(output)

    def recover_fills(self, since: datetime) -> tuple[ExecutionFill, ...]:
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since must be timezone-aware")
        return self._fills_from_history(since.astimezone(UTC), datetime.now(UTC), mark_seen=False)

    def _apply_reduce_only(self, intent: OrderIntent, request: dict[str, Any]) -> None:
        if not intent.reduce_only:
            return
        if intent.order_type is not OrderType.MARKET:
            raise PermanentBrokerError("MT5 reduce-only execution supports MARKET orders only")
        expected_type = (
            int(self.client.POSITION_TYPE_SELL)
            if intent.side is OrderSide.BUY
            else int(self.client.POSITION_TYPE_BUY)
        )
        candidates = [
            position
            for position in (self.client.positions_get(symbol=intent.symbol) or ())
            if int(_field(position, "type", -1)) == expected_type
            and self._belongs_to_adapter(position)
        ]
        candidates.sort(key=lambda item: int(_field(item, "ticket", 0)))
        quantity = float(request["volume"])
        position = next(
            (
                item
                for item in candidates
                if float(_field(item, "volume", 0.0)) + 1e-12 >= quantity
            ),
            None,
        )
        if position is None:
            raise PermanentBrokerError(
                "No opposite MT5 position can satisfy the reduce-only quantity"
            )
        request["position"] = int(_field(position, "ticket"))

    def _quote(self, symbol: str) -> Quote:
        raw = self.connection.symbol_tick(symbol)
        timestamp = record_time(raw)
        return Quote(
            symbol=symbol,
            bid=float(_field(raw, "bid")),
            ask=float(_field(raw, "ask")),
            timestamp=timestamp,
        )

    def _order_from_submission(
        self,
        intent: OrderIntent,
        request: dict[str, Any],
        result: Any,
        quote: Quote,
    ) -> BrokerOrder:
        classification = classification_from_result(self.client, result)
        order_ticket = _identifier(_field(result, "order", 0))
        deal_ticket = _identifier(_field(result, "deal", 0))
        broker_id = order_ticket if order_ticket not in {"", "0"} else deal_ticket
        if broker_id in {"", "0"}:
            broker_id = f"mt5-{intent.client_order_id}"
        requested = float(request["volume"])
        result_volume = float(_field(result, "volume", 0.0))
        done_code = int(getattr(self.client, "TRADE_RETCODE_DONE", -9999))
        partial_code = int(getattr(self.client, "TRADE_RETCODE_DONE_PARTIAL", -9998))
        placed_code = int(getattr(self.client, "TRADE_RETCODE_PLACED", -9997))
        if classification.retcode == partial_code:
            filled = min(max(result_volume, 0.0), requested)
            status = OrderStatus.PARTIALLY_FILLED
        elif classification.retcode == placed_code or (
            intent.order_type is not OrderType.MARKET and deal_ticket in {"", "0"}
        ):
            filled = 0.0
            status = OrderStatus.ACKNOWLEDGED
        elif classification.retcode == done_code:
            filled = requested
            status = OrderStatus.FILLED
        else:
            filled = min(max(result_volume, 0.0), requested)
            status = OrderStatus.FILLED if filled >= requested else OrderStatus.ACKNOWLEDGED
        average = None
        if filled > 0.0:
            raw_price = float(_field(result, "price", 0.0))
            average = raw_price or (quote.ask if intent.side is OrderSide.BUY else quote.bid)
        now = datetime.now(UTC)
        return BrokerOrder(
            broker_order_id=broker_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status=status,
            requested_quantity=requested,
            filled_quantity=filled,
            average_fill_price=average,
            submitted_at=now,
            updated_at=now,
            metadata=(
                ("comment", str(request["comment"])),
                ("magic", str(self.config.magic_number)),
                ("retcode", str(classification.retcode)),
            ),
        )

    def _order_from_raw(self, raw: Any) -> BrokerOrder:
        ticket = _identifier(_field(raw, "ticket"))
        initial = float(_field(raw, "volume_initial", _field(raw, "volume", 0.0)))
        remaining = float(_field(raw, "volume_current", initial))
        filled = min(max(initial - remaining, 0.0), initial)
        status = status_from_mt5_state(self.client, int(_field(raw, "state")))
        if status is OrderStatus.FILLED:
            filled = initial
        if status is OrderStatus.PARTIALLY_FILLED and not (0.0 < filled < initial):
            status = OrderStatus.ACKNOWLEDGED
        client_id = self._client_by_broker.get(ticket, f"mt5-{ticket}")
        average = float(_field(raw, "price_open", 0.0)) if filled > 0.0 else None
        submitted = record_time(raw)
        updated = record_time(raw)
        rejection = str(_field(raw, "comment", "rejected")) if status is OrderStatus.REJECTED else None
        order = BrokerOrder(
            broker_order_id=ticket,
            client_order_id=client_id,
            symbol=str(_field(raw, "symbol")),
            side=side_from_mt5_order_type(self.client, int(_field(raw, "type"))),
            order_type=order_type_from_mt5(self.client, int(_field(raw, "type"))),
            status=status,
            requested_quantity=initial,
            filled_quantity=filled,
            average_fill_price=average,
            submitted_at=submitted,
            updated_at=updated,
            rejection_reason=rejection,
            metadata=(
                ("comment", str(_field(raw, "comment", ""))),
                ("magic", str(_field(raw, "magic", 0))),
            ),
        )
        self._client_by_broker[ticket] = client_id
        return order

    def _queue_submission_fill(self, order: BrokerOrder, result: Any) -> None:
        deal = _identifier(_field(result, "deal", 0))
        if order.filled_quantity <= 0.0 or deal in {"", "0"}:
            return
        fill = ExecutionFill(
            execution_id=deal,
            broker_order_id=order.broker_order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.average_fill_price or 0.0,
            executed_at=order.updated_at,
            commission=0.0,
            metadata=(("source", "order_send"),),
        )
        self._seen_deals.add(fill.execution_id)
        self._pending_fills.append(fill)

    def _fills_from_history(
        self,
        start: datetime,
        end: datetime,
        *,
        mark_seen: bool,
    ) -> tuple[ExecutionFill, ...]:
        self.connection.ensure_connected()
        records = unique_by_ticket(self.client.history_deals_get(start, end) or ())
        output: list[ExecutionFill] = []
        buy_type = int(self.client.DEAL_TYPE_BUY)
        for raw in records:
            if not self._belongs_to_adapter(raw):
                continue
            execution_id = _identifier(_field(raw, "ticket"))
            if execution_id in self._seen_deals:
                continue
            broker_id = _identifier(_field(raw, "order", _field(raw, "position_id", 0)))
            client_id = self._client_by_broker.get(broker_id, f"mt5-{broker_id}")
            fill = ExecutionFill(
                execution_id=execution_id,
                broker_order_id=broker_id,
                client_order_id=client_id,
                symbol=str(_field(raw, "symbol")),
                side=(
                    OrderSide.BUY
                    if int(_field(raw, "type", buy_type)) == buy_type
                    else OrderSide.SELL
                ),
                quantity=float(_field(raw, "volume")),
                price=float(_field(raw, "price")),
                executed_at=record_time(raw),
                commission=abs(float(_field(raw, "commission", 0.0))),
                metadata=(
                    ("deal_entry", str(_field(raw, "entry", ""))),
                    ("magic", str(_field(raw, "magic", 0))),
                ),
            )
            output.append(fill)
            if mark_seen:
                self._seen_deals.add(execution_id)
        output.sort(key=lambda item: (item.executed_at, item.execution_id))
        return tuple(output)

    def _belongs_to_adapter(self, raw: Any) -> bool:
        if self.config.include_external_orders:
            return True
        return int(_field(raw, "magic", 0)) == self.config.magic_number

    def _cache_order(self, order: BrokerOrder) -> None:
        self._orders[order.broker_order_id] = order
        self._client_by_broker[order.broker_order_id] = order.client_order_id


__all__ = ["MT5BrokerAdapter", "MT5ExecutionConfig"]
