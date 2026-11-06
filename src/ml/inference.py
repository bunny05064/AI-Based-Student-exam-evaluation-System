"""High-level inference helpers used by the evaluation service.

These functions are intentionally thin: they wire the loaded ModelManager
into the ConceptExtractor and optional correctness classifier, keeping the
pure-domain logic (rubric, feedback, persistence) in ``app.services``.
"""

from typing import List, Optional

from src.ml.concept_extractor import ConceptExtractor
from src.ml.model import ModelManager, model_manager


def match_concepts(
    question: str,
    answer: str,
    expected_concepts: List[dict],
    manager: Optional[ModelManager] = None,
) -> List[dict]:
    """Return per-concept match dicts:
    {"concept", "status", "evidence", "confidence", "weight"}.
    """
    mgr = manager or model_manager
    if not mgr.loaded:
        raise RuntimeError("ModelManager must be loaded before inference.")
    correct, partial = mgr.thresholds()
    extractor = ConceptExtractor(
        embedding_provider=mgr.provider,
        correct_threshold=correct,
        partial_threshold=partial,
    )
    matches = extractor.evaluate(question, answer, expected_concepts)
    return [m.to_dict() for m in matches]


def classify_answer(
    question: str,
    answer: str,
    manager: Optional[ModelManager] = None,
) -> Optional[dict]:
    """Optional overall correctness prediction (None in development mode).

    `manager` must be honoured: falling back to the process-wide singleton here
    would silently pair an injected model with a different classifier.
    """
    mgr = manager or model_manager
    if not mgr.loaded:
        raise RuntimeError("ModelManager must be loaded before inference.")
    result = mgr.classify(question, answer)
    if result is None:
        return None
    return result.to_dict()