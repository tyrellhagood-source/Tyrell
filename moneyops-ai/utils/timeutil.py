"""Timezone helpers. All timestamps are stored in UTC."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    # ESPN sometimes omits seconds, e.g. 2026-10-20T23:30+00:00
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_local(dt: datetime | None, tz: str) -> str:
    if dt is None:
        return "n/a"
    return ensure_utc(dt).astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M %Z")


def minutes_between(later: datetime, earlier: datetime) -> float:
    return (ensure_utc(later) - ensure_utc(earlier)).total_seconds() / 60.0
