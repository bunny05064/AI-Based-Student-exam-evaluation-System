"""OCR abstraction so the pipeline never depends on a specific engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OCRResult:
    text: str
    confidence: float
    pages: int
    engine: str
    metadata: Dict = field(default_factory=dict)

    def _as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "pages": self.pages,
            "engine": self.engine,
            "metadata": self.metadata,
        }


class OCREngine(ABC):
    """Interface for any OCR backend."""

    name: str = "base"

    @abstractmethod
    def extract_text(self, data: bytes, is_pdf: bool = False) -> OCRResult:
        """Extract text from raw image bytes (or PDF pages when supported)."""

    @classmethod
    def is_available(cls) -> bool:
        """True when this engine's dependencies are actually installed.

        Checked at wiring time so the app can pick a working engine (or fail
        loudly) instead of 500-ing on the first upload.
        """
        return True

    def __bool__(self) -> bool:
        return True


def has_ready_engine(engine_class) -> bool:
    """Backwards-compatible alias for `engine_class.is_available()`."""
    try:
        return bool(engine_class.is_available())
    except Exception:  # noqa: BLE001
        return False