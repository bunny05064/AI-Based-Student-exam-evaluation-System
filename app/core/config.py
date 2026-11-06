"""Core configuration for the application.

Single source of truth for runtime settings. Values are read from the
environment (optionally via a `.env` file) with sensible defaults so the
project runs out of the box in development mode.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

_logger = logging.getLogger("app.config")


class Env:
    """Read-only view over an environment mapping.

    Never mutates `os.environ`: an explicit `overrides` mapping is layered on
    top of the process environment for the lifetime of this object only, so
    building settings (e.g. in tests) cannot leak into unrelated code.
    """

    def __init__(self, overrides: Optional[Mapping[str, str]] = None):
        self._values: Dict[str, str] = dict(os.environ)
        if overrides:
            self._values.update({k: str(v) for k, v in overrides.items()})

    def str(self, name: str, default: str) -> str:
        value = self._values.get(name)
        return value.strip() if value is not None and value.strip() != "" else default

    def bool(self, name: str, default: bool) -> bool:
        value = self._values.get(name)
        if value is None or value.strip() == "":
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def int(self, name: str, default: int) -> int:
        raw = self._values.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw.strip())
        except (TypeError, ValueError):
            _logger.warning(
                "Invalid integer for %s=%r - falling back to %r", name, raw, default
            )
            return default

    def float(self, name: str, default: float) -> float:
        raw = self._values.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return float(raw.strip())
        except (TypeError, ValueError):
            _logger.warning(
                "Invalid float for %s=%r - falling back to %r", name, raw, default
            )
            return default

    def float_opt(self, name: str) -> Optional[float]:
        raw = self._values.get(name)
        if raw is None or raw.strip() == "":
            return None
        try:
            return float(raw.strip())
        except (TypeError, ValueError):
            _logger.warning("Ignoring invalid float for %s=%r", name, raw)
            return None

    def csv(self, name: str, default: List[str]) -> List[str]:
        value = self._values.get(name)
        if not value or not value.strip():
            return list(default)
        return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    """Runtime settings. All defaults are safe for local development."""

    flask_env: str = "development"
    secret_key: str = "dev-secret-change-me"
    database_url: str = "sqlite:///app.db"

    # Model / ML
    model_mode: str = "development"  # "development" | "production"
    model_path: str = "models/model"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fallback: str = "tfidf"
    model_preload: bool = False

    # OCR
    ocr_engine: str = "tesseract"  # "tesseract" | "easyocr" | "mock"
    ocr_min_confidence: float = 0.0

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Validation limits
    max_upload_mb: int = 10
    max_answer_chars: int = 10000
    max_question_chars: int = 2000
    max_score_default: int = 10
    max_score_allowed: int = 100
    max_pdf_pages: int = 20

    # Concept matching thresholds (cosine similarity in [0, 1]).
    # None -> provider-specific defaults (TF-IDF scores sit lower than SBERT).
    concept_threshold_correct: Optional[float] = None
    concept_threshold_partial: Optional[float] = None

    # Rubric scoring
    partial_credit_factor: float = 0.5
    incorrect_penalty: float = 0.0

    # Web
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Data pipeline
    dataset_raw_dir: str = "data/raw"
    dataset_interim_dir: str = "data/interim"
    dataset_processed_dir: str = "data/processed"
    test_seed: int = 42

    # ------------------------------------------------------------------ #
    def abs_path(self, relative: str) -> Path:
        """Resolve a project-relative path to an absolute one (no hardcoding)."""
        path = Path(relative)
        if path.is_absolute():
            return path
        return BASE_DIR / path

    @property
    def log_path(self) -> "Path":
        return self.abs_path(self.log_dir)

    @property
    def model_abs_path(self) -> "Path":
        return self.abs_path(self.model_path)

    @property
    def is_production(self) -> bool:
        return self.flask_env == "production" and self.model_mode == "production"

    @property
    def development_model(self) -> bool:
        """True when no trained/transformer model should be expected."""
        return self.model_mode == "development"

    @property
    def engine_options(self) -> dict:
        """Engine options tuned for long-lived server processes.

        `pool_pre_ping` avoids the "server closed the connection" 500s that a
        long-idle Postgres connection produces behind gunicorn workers.
        """
        if self.database_url.startswith("sqlite"):
            return {}
        return {"pool_pre_ping": True, "pool_recycle": 280, "pool_size": 5}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from the process env, with optional local overrides.

        `environ` is *not* written back to `os.environ` - it only shadows
        values for this call.
        """
        env = Env(environ)
        return cls(
            flask_env=env.str("FLASK_ENV", "development"),
            secret_key=env.str("SECRET_KEY", "dev-secret-change-me"),
            database_url=env.str("DATABASE_URL", "sqlite:///app.db"),
            model_mode=env.str("MODEL_MODE", "development"),
            model_path=env.str("MODEL_PATH", "models/model"),
            embedding_model=env.str(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_fallback=env.str("EMBEDDING_FALLBACK", "tfidf"),
            model_preload=env.bool("MODEL_PRELOAD", False),
            ocr_engine=env.str("OCR_ENGINE", "tesseract"),
            ocr_min_confidence=env.float("OCR_MIN_CONFIDENCE", 0.0),
            log_level=env.str("LOG_LEVEL", "INFO"),
            log_dir=env.str("LOG_DIR", "logs"),
            max_upload_mb=env.int("MAX_UPLOAD_MB", 10),
            max_answer_chars=env.int("MAX_ANSWER_CHARS", 10000),
            max_question_chars=env.int("MAX_QUESTION_CHARS", 2000),
            max_score_default=env.int("MAX_SCORE_DEFAULT", 10),
            max_score_allowed=env.int("MAX_SCORE_ALLOWED", 100),
            max_pdf_pages=env.int("MAX_PDF_PAGES", 20),
            concept_threshold_correct=env.float_opt("CONCEPT_THRESHOLD_CORRECT"),
            concept_threshold_partial=env.float_opt("CONCEPT_THRESHOLD_PARTIAL"),
            partial_credit_factor=env.float("PARTIAL_CREDIT_FACTOR", 0.5),
            incorrect_penalty=env.float("INCORRECT_PENALTY", 0.0),
            cors_origins=env.csv("CORS_ORIGINS", ["*"]),
            dataset_raw_dir=env.str("DATASET_RAW_DIR", "data/raw"),
            dataset_interim_dir=env.str("DATASET_INTERIM_DIR", "data/interim"),
            dataset_processed_dir=env.str("DATASET_PROCESSED_DIR", "data/processed"),
            test_seed=env.int("TEST_SEED", 42),
        )


def load_settings(overrides: Optional[Mapping[str, str]] = None) -> Settings:
    """Load dotenv once, then build Settings from the environment."""
    load_dotenv(BASE_DIR / ".env")
    return Settings.from_env(overrides)


def flask_config(settings: Settings) -> dict:
    """Map Settings to a Flask app.config dictionary."""
    return {
        "SECRET_KEY": settings.secret_key,
        "FLASK_DEBUG": settings.flask_env == "development",
        "DATABASE_URL": settings.database_url,
        "SQLALCHEMY_DATABASE_URI": settings.database_url,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_ENGINE_OPTIONS": settings.engine_options,
        "MAX_CONTENT_LENGTH": settings.max_upload_mb * 1024 * 1024,
        "SETTINGS": settings,
    }


def public_limits(settings: Settings) -> dict:
    """Client-facing limits, so the UI never hardcodes what the server enforces."""
    return {
        "max_upload_mb": settings.max_upload_mb,
        "max_answer_chars": settings.max_answer_chars,
        "max_question_chars": settings.max_question_chars,
        "max_score_default": settings.max_score_default,
        "max_score_allowed": settings.max_score_allowed,
        "ocr_engine": settings.ocr_engine,
        "model_mode": settings.model_mode,
    }