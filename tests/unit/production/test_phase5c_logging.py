import json
import logging

from fxbot.production.logging import JsonFormatter


def test_json_formatter_includes_context() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        "fxbot.test",
        logging.INFO,
        __file__,
        1,
        "hello %s",
        ("world",),
        None,
    )
    record.order_id = "123"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello world"
    assert payload["context"]["order_id"] == "123"
