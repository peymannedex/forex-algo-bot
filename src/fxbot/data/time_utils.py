"""Timestamp parsing and UTC normalization utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.parser import isoparse


class TimestampParseError(ValueError):
    """Raised when a source timestamp cannot be normalized safely."""


def parse_timestamp(
    value: object,
    *,
    timestamp_format: str | None = None,
    naive_timezone: str = "UTC",
) -> datetime:
    """Parse a timestamp and return a timezone-aware UTC ``datetime``.

    Args:
        value: String, POSIX timestamp, or ``datetime``.
        timestamp_format: Optional explicit ``datetime.strptime`` format.
        naive_timezone: IANA timezone assigned to timestamps lacking an offset.
            This must describe the source feed's timezone; it is not a display
            preference.
    """

    try:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, (int, float)):
            parsed = datetime.fromtimestamp(float(value), tz=UTC)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise TimestampParseError("timestamp cannot be empty")
            parsed = (
                datetime.strptime(text, timestamp_format)
                if timestamp_format is not None
                else isoparse(text)
            )
        else:
            raise TimestampParseError(f"Unsupported timestamp type: {type(value)!r}")

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(naive_timezone))
            except ZoneInfoNotFoundError as exc:
                raise TimestampParseError(
                    f"Unknown naive timestamp timezone: {naive_timezone!r}"
                ) from exc
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, TimestampParseError):
            raise
        raise TimestampParseError(f"Cannot parse timestamp {value!r}: {exc}") from exc
