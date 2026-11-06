"""Production-style logging setup.

- RotatingFileHandler for the application log (append.log)
- Separate rotating error log (error.log) that only receives ERROR+
- Console handler (StreamHandler) for local development
- A ``request_id`` context variable so concurrent requests can be traced

Example log line:
2026-09-25 10:32:12 | INFO | app.routes.evaluation | request=abc123 | Evaluation started | id=xyz
"""

import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | request=%(request_id)s | %(message)s"
)

# Context variable carrying the current request ID for log correlation.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attach the active request_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_configured_log_files: Dict[str, Path] = {}


def _make_handler(
    path: Path, level: int, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 5
) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())
    return handler


def setup_logging(
    log_dir: Path, level: str = "INFO", verbose_console: bool = True
) -> None:
    """Configure the root logger. Safe to call multiple times (idempotent)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    level = getattr(logging, level.upper(), logging.INFO)

    # Avoid duplicate handlers when the app is created more than once (tests).
    if getattr(root, "_atr_configured", False):
        return
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose_console else level)
    console.setFormatter(formatter)
    console.addFilter(RequestIdFilter())
    root.addHandler(console)

    app_handler = _make_handler(log_dir / "app.log", logging.DEBUG)
    error_handler = _make_handler(log_dir / "error.log", logging.ERROR)
    root.addHandler(app_handler)
    root.addHandler(error_handler)

    root._atr_configured = True
    _configured_log_files["app"] = log_dir / "app.log"
    _configured_log_files["error"] = log_dir / "error.log"


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger (all writing through the root config)."""
    return logging.getLogger(name)