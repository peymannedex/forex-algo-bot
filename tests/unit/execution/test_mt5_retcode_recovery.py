from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from fxbot.execution.broker import (
    PermanentBrokerError,
    TransientBrokerError,
    UnknownSubmissionError,
)
from fxbot.execution.mt5_recovery import (
    MT5RetcodeCategory,
    classification_from_result,
    classify_mt5_retcode,
    raise_for_mt5_result,
    record_time,
    unique_by_ticket,
)


def test_success_retcode(client: Any) -> None:
    result = classify_mt5_retcode(client, client.TRADE_RETCODE_DONE)

    assert result.success
    assert result.category is MT5RetcodeCategory.SUCCESS


def test_transient_retcode(client: Any) -> None:
    result = classify_mt5_retcode(client, client.TRADE_RETCODE_REQUOTE)

    assert result.category is MT5RetcodeCategory.TRANSIENT


def test_timeout_is_unknown_submission(client: Any) -> None:
    result = classify_mt5_retcode(client, client.TRADE_RETCODE_TIMEOUT)

    assert result.category is MT5RetcodeCategory.UNKNOWN_SUBMISSION


def test_unknown_code_is_permanent(client: Any) -> None:
    result = classify_mt5_retcode(client, 99999)

    assert result.category is MT5RetcodeCategory.PERMANENT


def test_none_result_unknown(client: Any) -> None:
    result = classification_from_result(client, None)

    assert result.category is MT5RetcodeCategory.UNKNOWN_SUBMISSION


def test_raise_for_result(client: Any) -> None:
    with pytest.raises(TransientBrokerError):
        raise_for_mt5_result(
            client,
            SimpleNamespace(
                retcode=client.TRADE_RETCODE_REQUOTE,
                comment="retry",
            ),
            "send",
        )

    with pytest.raises(UnknownSubmissionError):
        raise_for_mt5_result(
            client,
            SimpleNamespace(
                retcode=client.TRADE_RETCODE_TIMEOUT,
                comment="timeout",
            ),
            "send",
        )

    with pytest.raises(PermanentBrokerError):
        raise_for_mt5_result(
            client,
            SimpleNamespace(
                retcode=client.TRADE_RETCODE_INVALID_VOLUME,
                comment="bad",
            ),
            "send",
        )


def test_record_time_prefers_milliseconds() -> None:
    record = SimpleNamespace(time=1, time_msc=1704067200123)

    assert record_time(record) == datetime.fromtimestamp(
        1704067200.123,
        tz=UTC,
    )


def test_unique_records_sorted_and_deduplicated() -> None:
    records = [
        SimpleNamespace(ticket=2, time=2),
        SimpleNamespace(ticket=1, time=1),
        SimpleNamespace(ticket=2, time=3),
    ]

    assert [item.ticket for item in unique_by_ticket(records)] == [1, 2]
