"""MT5 return-code classification and raw history recovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fxbot.execution.broker import (
    PermanentBrokerError,
    TransientBrokerError,
    UnknownSubmissionError,
)


class MT5RetcodeCategory(StrEnum):
    SUCCESS = "success"
    TRANSIENT = "transient"
    UNKNOWN_SUBMISSION = "unknown_submission"
    PERMANENT = "permanent"


@dataclass(frozen=True, slots=True)
class MT5RetcodeClassification:
    retcode: int
    category: MT5RetcodeCategory
    name: str
    message: str

    @property
    def success(self) -> bool:
        return self.category is MT5RetcodeCategory.SUCCESS


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _constants(client: Any, names: tuple[str, ...]) -> dict[int, str]:
    output: dict[int, str] = {}
    for name in names:
        value = getattr(client, name, None)
        if value is not None:
            output[int(value)] = name
    return output


def classify_mt5_retcode(client: Any, retcode: int, message: str = "") -> MT5RetcodeClassification:
    """Classify an MT5 trade return code by safe retry semantics."""

    success = _constants(
        client,
        (
            "TRADE_RETCODE_DONE",
            "TRADE_RETCODE_PLACED",
            "TRADE_RETCODE_DONE_PARTIAL",
            "TRADE_RETCODE_NO_CHANGES",
        ),
    )
    unknown = _constants(
        client,
        (
            "TRADE_RETCODE_TIMEOUT",
            "TRADE_RETCODE_CONNECTION",
        ),
    )
    transient = _constants(
        client,
        (
            "TRADE_RETCODE_REQUOTE",
            "TRADE_RETCODE_PRICE_CHANGED",
            "TRADE_RETCODE_PRICE_OFF",
            "TRADE_RETCODE_TOO_MANY_REQUESTS",
            "TRADE_RETCODE_LOCKED",
        ),
    )
    code = int(retcode)
    if code in success or code == 0:
        category = MT5RetcodeCategory.SUCCESS
        name = success.get(code, "CHECK_OK")
    elif code in unknown:
        category = MT5RetcodeCategory.UNKNOWN_SUBMISSION
        name = unknown[code]
    elif code in transient:
        category = MT5RetcodeCategory.TRANSIENT
        name = transient[code]
    else:
        category = MT5RetcodeCategory.PERMANENT
        name = next(
            (
                attr
                for attr in dir(client)
                if attr.startswith("TRADE_RETCODE_") and getattr(client, attr, object()) == code
            ),
            f"TRADE_RETCODE_{code}",
        )
    return MT5RetcodeClassification(code, category, name, message.strip())


def classification_from_result(client: Any, result: Any) -> MT5RetcodeClassification:
    if result is None:
        return MT5RetcodeClassification(
            -1,
            MT5RetcodeCategory.UNKNOWN_SUBMISSION,
            "NO_RESULT",
            "MT5 returned no trade result",
        )
    return classify_mt5_retcode(
        client,
        int(_field(result, "retcode", -1)),
        str(_field(result, "comment", "")),
    )


def raise_for_mt5_result(client: Any, result: Any, operation: str) -> None:
    """Raise the Phase 5A broker error matching an MT5 result."""

    classification = classification_from_result(client, result)
    if classification.success:
        return
    detail = f"{operation} failed: {classification.name} ({classification.retcode})"
    if classification.message:
        detail += f": {classification.message}"
    if classification.category is MT5RetcodeCategory.UNKNOWN_SUBMISSION:
        raise UnknownSubmissionError(detail)
    if classification.category is MT5RetcodeCategory.TRANSIENT:
        raise TransientBrokerError(detail)
    raise PermanentBrokerError(detail)


def record_time(record: Any) -> datetime:
    """Extract the most precise UTC timestamp available on an MT5 record."""

    for name in ("time_msc", "time_done_msc", "time_setup_msc"):
        value = _field(record, name, None)
        if value not in (None, 0):
            return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    for name in ("time", "time_done", "time_setup"):
        value = _field(record, name, None)
        if value not in (None, 0):
            return datetime.fromtimestamp(float(value), tz=UTC)
    return datetime.now(UTC)


def unique_by_ticket(records: Any) -> tuple[Any, ...]:
    """Return raw MT5 records once, ordered by time and ticket."""

    output: list[Any] = []
    seen: set[int] = set()
    for record in records or ():
        ticket = int(_field(record, "ticket", 0))
        if ticket in seen:
            continue
        seen.add(ticket)
        output.append(record)
    output.sort(key=lambda item: (record_time(item), int(_field(item, "ticket", 0))))
    return tuple(output)


__all__ = [
    "MT5RetcodeCategory",
    "MT5RetcodeClassification",
    "classification_from_result",
    "classify_mt5_retcode",
    "raise_for_mt5_result",
    "record_time",
    "unique_by_ticket",
]
