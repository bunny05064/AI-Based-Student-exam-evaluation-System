"""ModelManager: one process-lifetime singleton for all ML artifacts.

The heavy components (embedding provider, correctness classifier) are loaded
once and reused across requests. Tests can inject fakes via ``load(..., provider=,
classifier=)`` or call ``reset()``.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from src.ml.classifier import (
    ClassificationResult,
    CorrectnessClassifier,
    load_classifier,
)
from src.ml.concept_extractor import resolve_thresholds
from src.ml.embeddings import EmbeddingProvider, resolve_embedding_provider

logger = logging.getLogger("src.ml.model")


class ModelManager:
    """Holds the active embedding provider and correctness classifier."""

    def __init__(self) -> None:
        self._provider: Optional[EmbeddingProvider] = None
        self._classifier: Optional[CorrectnessClassifier] = None
        self._settings = None
        self._loaded = False
        self.mode_label: str = "development"

    # ------------------------------------------------------------------ #
    def configure(self, settings) -> None:
        """Record the configured settings WITHOUT loading heavy artifacts.

        ``model_info()`` reports the *configured* mode before ``load()`` runs
        (the normal case with ``MODEL_PRELOAD=false``). Without this the API
        returned ``model_mode="unknown"`` and the home page mislabelled a
        development deployment as production mode.
        """
        self._settings = settings

    def load(
        self,
        settings,
        provider: Optional[EmbeddingProvider] = None,
        classifier: Optional[CorrectnessClassifier] = None,
    ) -> None:
        self._settings = settings
        if provider is not None:
            self._provider = provider
        else:
            try:
                self._provider = resolve_embedding_provider(settings)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Could not resolve embedding provider")
                raise exc

        if classifier is not None:
            self._classifier = classifier
        else:
            self._classifier = load_classifier(
                str(settings.model_abs_path), settings.model_mode
            )

        self.mode_label = settings.model_mode
        self._loaded = True
        logger.info(
            "ModelManager loaded | provider=%s classifier=%s mode=%s",
            self.provider_name,
            self.classifier_name,
            self.mode_label,
        )

    def reset(self) -> None:
        self._provider = None
        self._classifier = None
        self._settings = None
        self._loaded = False
        self.mode_label = "development"

    # ------------------------------------------------------------------ #
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def provider(self) -> EmbeddingProvider:
        if not self._loaded or self._provider is None:
            raise RuntimeError("ModelManager is not loaded; call load() first.")
        return self._provider

    @property
    def provider_name(self) -> str:
        return self._provider.name if self._provider is not None else "unavailable"

    @property
    def classifier_name(self) -> str:
        if self._classifier is None:
            return "none"
        probe = self._classifier.predict("", "")
        return probe.model_name

    def classify(self, question: str, answer: str) -> Optional[ClassificationResult]:
        if self._classifier is None:
            return None
        return self._classifier.predict(question, answer)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.provider.encode_texts(texts)

    def thresholds(self) -> Tuple[float, float]:
        correct, partial = resolve_thresholds(
            self._settings, self.provider_name
        )
        return correct, partial

    def model_info(self) -> dict:
        """Summary used by GET /api/v1/model/info.

        Reports the *configured* mode even before `load()` runs. The previous
        version read `self._settings`, which is None until the first lazy
        load, so a production deployment with `MODEL_PRELOAD=false` advertised
        itself as "Development model mode" on the home page.
        """
        configured_mode = getattr(self._settings, "model_mode", None) if self._settings else None
        info: dict = {
            "model_mode": configured_mode or "unknown",
            "loaded": self._loaded,
            "embedding_provider": self.provider_name if self._loaded else "not-loaded",
            "classifier": self.classifier_name if self._loaded else "not-loaded",
        }
        if self._settings is not None:
            info["embedding_model_configured"] = self._settings.embedding_model
            info["model_path"] = str(self._settings.model_path)
        if configured_mode and not self._loaded:
            info["note"] = (
                "Model is configured but not loaded yet; it loads on first use "
                "(set MODEL_PRELOAD=true to load at startup)."
            )
        return info


# Process-lifetime singleton.
model_manager = ModelManager()