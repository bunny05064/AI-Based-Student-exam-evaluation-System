"""Thin root-level re-export so `from config import load_settings` works.

The real implementation lives in app/core/config.py (single source of truth).
Settings are built per application instance rather than cached at import time,
so importing this module never has the side effect of reading `.env`.
"""

from app.core.config import (
    BASE_DIR,
    Settings,
    flask_config,
    load_settings,
    public_limits,
)

__all__ = ["BASE_DIR", "Settings", "flask_config", "load_settings", "public_limits"]