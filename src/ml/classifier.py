"""Correctness classifiers.

- ``NoOpClassifier``: development mode - never claims a real prediction.
- ``MLBaselineClassifier``: sklearn TF-IDF + logistic regression (trainable
  without torch; Experiment 1 / 2 baseline).
- ``TransformerClassifier``: a trained Hugging Face encoder loaded from disk
  (Experiment 3).

All share the ``predict(question, answer)`` interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.exceptions import ModelLoadError
from src.ml.labels import LABEL2ID


@dataclass
class ClassificationResult:
    label: Optional[str]  # "correct" | "incorrect" | None
    confidence: float
    model_name: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "model_name": self.model_name,
        }


class CorrectnessClassifier(ABC):
    @abstractmethod
    def predict(self, question: str, answer: str) -> ClassificationResult:
        """Predict overall answer correctness (may return None if unavailable)."""


class NoOpClassifier(CorrectnessClassifier):
    """Used in development mode: returns None - never fakes a prediction."""

    def predict(self, question: str, answer: str) -> ClassificationResult:
        return ClassificationResult(label=None, confidence=0.0, model_name="none")


class MLBaselineClassifier(CorrectnessClassifier):
    """Loads a pre-trained joblib pipeline produced by scripts/train.py."""

    def __init__(self, pipeline_path: str) -> None:
        import joblib  # noqa: PLC0415

        try:
            self._pipeline = joblib.load(pipeline_path)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Could not load baseline model at {pipeline_path}.") from exc

    def predict(self, question: str, answer: str) -> ClassificationResult:
        try:
            proba = self._pipeline.predict_proba([f"{question} [SEP] {answer}"])[0]
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError("Baseline classifier inference failed.") from exc
        classes = list(self._pipeline.classes_)
        idx = int(np.argmax(proba))
        return ClassificationResult(
            label=str(classes[idx]),
            confidence=float(proba[idx]),
            model_name="baseline-logistic",
        )


class TransformerClassifier(CorrectnessClassifier):
    """Loads a trained encoder classifier (transformers + torch, lazy)."""

    def __init__(self, model_dir: str) -> None:
        from pathlib import Path

        self.model_dir = Path(model_dir)
        try:
            # Importing transformers is enough: it imports torch as a hard
            # dependency, so no separate torch probe is needed here.
            from transformers import (  # noqa: PLC0415
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ModelLoadError(
                "torch/transformers not installed; install requirements-ml.txt."
            ) from exc

        try:
            self._config = AutoConfig.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            )
            self._model.eval()
            # Restored once here, not per request: re-parsing the tokenizer on
            # every inference was a full disk/HF-cache hit under each call.
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._label_map = getattr(self._config, "label2id", None) or dict(LABEL2ID)
            self._id2label = {v: k for k, v in self._label_map.items()}
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load transformer model from {model_dir}.") from exc

    def predict(self, question: str, answer: str) -> ClassificationResult:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - guarded at load time
            raise ModelLoadError("torch is not installed.") from exc
        try:
            inputs = self._tokenizer(
                question, answer, truncation=True, max_length=256, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            idx = int(np.argmax(probs))
            return ClassificationResult(
                label=self._id2label.get(idx, "correct"),
                confidence=float(probs[idx]),
                model_name="transformer-encoder",
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError("Transformer inference failed.") from exc


def load_classifier(model_dir: str, mode: str) -> CorrectnessClassifier:
    """Load the trained classifier bundle.

    Development mode falls back to ``NoOpClassifier`` (which never claims a
    prediction). Production mode is **strict**: a missing or unreadable bundle
    raises ``ModelLoadError`` rather than silently degrading to TF-IDF +
    NoOp and serving confident-looking scores from a model that was never
    trained.
    """
    import json
    from pathlib import Path

    if mode == "development":
        return NoOpClassifier()

    directory = Path(model_dir)
    config_path = directory / "config.json"
    if not config_path.exists():
        raise ModelLoadError(
            f"MODEL_MODE=production but no trained model bundle at {directory}. "
            "Run `python scripts/train.py` first, or set MODEL_MODE=development "
            "to use the classical TF-IDF pipeline (labelled as development mode)."
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Model config at {config_path} is not readable JSON.") from exc

    kind = config.get("model_type", "baseline")
    if kind in {"baseline", "logistic", "tfidf"}:
        joblib_path = directory / "model.joblib"
        if not joblib_path.exists():
            raise ModelLoadError(f"Model bundle {directory} is missing {joblib_path.name}.")
        return MLBaselineClassifier(str(joblib_path))
    if kind == "transformer":
        return TransformerClassifier(str(directory))
    raise ModelLoadError(f"Unknown model_type {kind!r} in {config_path}.")