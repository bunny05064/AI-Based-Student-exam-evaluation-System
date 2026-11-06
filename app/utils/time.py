"""Timezone-safe UTC helpers (avoid deprecated datetime.utcnow)."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC now, matching the old datetime.utcnow() semantics so values
    stay compatible with SQLAlchemy db.DateTime columns (SQLite stores naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)