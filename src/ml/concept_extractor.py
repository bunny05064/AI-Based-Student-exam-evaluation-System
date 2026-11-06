"""Concept extraction & semantic matching - the explainability core.

Given a question, a student answer and a set of expected concepts (from the
rubric, a built-in registry or auto-generated keyphrases), each concept is
labelled as one of:

    correct | partially_correct | missing | incorrect

Matching uses semantic (cosine) similarity between concept+variant embeddings
and the answer's sentence embeddings - NOT naive substring search. Each result
carries the *evidence sentence* and a confidence (the measured cosine
similarity), which is the exact text-level reason behind a score.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from app.utils.text import segment_sentences
from src.ml.embeddings import EmbeddingProvider

VALID_STATUSES = ("correct", "partially_correct", "missing", "incorrect")


@dataclass
class ConceptMatch:
    concept: str
    status: str
    evidence: str
    confidence: float
    similarity: float
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "status": self.status,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 4),
            "weight": float(self.weight),
        }


class ConceptExtractor:
    """Match expected concepts against an answer using semantic similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        correct_threshold: float = 0.60,
        partial_threshold: float = 0.45,
    ) -> None:
        if not (0.0 <= partial_threshold <= correct_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 <= partial <= correct <= 1."
            )
        self.provider = embedding_provider
        self.correct_threshold = correct_threshold
        self.partial_threshold = partial_threshold
        self._sentences: List[str] = []

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        question: str,
        answer: str,
        expected_concepts: List[dict],
    ) -> List[ConceptMatch]:
        """Return a ConceptMatch for every expected concept.

        ``expected_concepts`` items: {"name", "weight", "synonyms",
        "incorrect_phrases"}. The question text is passed so future
        question-conditioned models can use it; sentence-vs-concept matching
        already works in the shared vector space of one encoding call.
        """
        self._sentences = segment_sentences(answer)
        if not self._sentences:
            return [
                ConceptMatch(
                    concept=c.get("name", ""),
                    status="missing",
                    evidence="",
                    confidence=0.0,
                    similarity=0.0,
                    weight=float(c.get("weight", 1.0)),
                )
                for c in expected_concepts
                if c.get("name")
            ]

        rows, row_meta, concept_index = self._build_variant_rows(expected_concepts)
        if not rows:
            return [
                ConceptMatch(
                    concept=c.get("name", ""),
                    status="missing",
                    evidence="",
                    confidence=0.0,
                    similarity=0.0,
                    weight=float(c.get("weight", 1.0)),
                )
                for c in expected_concepts
                if c.get("name")
            ]

        self._sim_matrix = self.provider.similarity_matrix(rows, self._sentences)

        results: List[ConceptMatch] = []
        for ci, concept in enumerate(concept_index):
            name = concept["name"]
            weight = float(concept.get("weight", 1.0))
            if weight <= 0:
                continue

            positive_rows = [r for r, (kind, k) in enumerate(row_meta) if kind == "pos" and k == ci]
            negative_rows = [r for r, (kind, k) in enumerate(row_meta) if kind == "neg" and k == ci]

            status, best_sim, best_sentence = self._classify(
                positive_rows, negative_rows
            )
            results.append(
                ConceptMatch(
                    concept=name,
                    status=status,
                    evidence=best_sentence,
                    confidence=float(np.clip(best_sim, 0.0, 1.0)),
                    similarity=float(best_sim),
                    weight=weight,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    def _build_variant_rows(
        self, expected_concepts: List[dict]
    ) -> Tuple[List[str], List[tuple], List[dict]]:
        rows: List[str] = []
        row_meta: List[tuple] = []
        concept_index: List[dict] = []
        for ci, concept in enumerate(expected_concepts):
            name = (concept.get("name") or "").strip()
            if not name:
                continue
            concept_index.append({**concept, "name": name})
            variants = [name] + [s for s in (concept.get("synonyms") or []) if s]
            for v in variants:
                rows.append(v)
                row_meta.append(("pos", ci))
            for phrase in concept.get("incorrect_phrases") or []:
                if phrase:
                    rows.append(phrase)
                    row_meta.append(("neg", ci))
        return rows, row_meta, concept_index

    def _classify(
        self,
        positive_rows: List[int],
        negative_rows: List[int],
    ) -> Tuple[str, float, str]:
        """Pick the concept status and the best evidence sentence."""
        if not positive_rows:
            return "missing", 0.0, ""

        sims_pos = self._sim_matrix[np.ix_(positive_rows, range(self._sim_matrix.shape[1]))]
        best_flat = int(np.argmax(sims_pos))
        best_var_local, best_sent = np.unravel_index(
            best_flat, sims_pos.shape
        )  # type: ignore[misc]
        best_sim = float(sims_pos[best_var_local, best_sent])

        # Contradiction check: an explicit "this is false" phrase matches.
        if negative_rows:
            sims_neg = self._sim_matrix[np.ix_(negative_rows, range(self._sim_matrix.shape[1]))]
            neg_best_flat = int(np.argmax(sims_neg))
            neg_var_local, neg_sent = np.unravel_index(
                neg_best_flat, sims_neg.shape
            )  # type: ignore[misc]
            neg_best = float(sims_neg[neg_var_local, neg_sent])
            if neg_best >= self.correct_threshold:
                return "incorrect", neg_best, self._evidence_at(_sent=neg_sent)

        if best_sim >= self.correct_threshold:
            return "correct", best_sim, self._evidence_at(_sent=best_sent)
        if best_sim >= self.partial_threshold:
            return "partially_correct", best_sim, self._evidence_at(_sent=best_sent)
        return "missing", best_sim, self._evidence_at(_sent=best_sent)

    def _evidence_at(self, _sent: int) -> str:
        if 0 <= _sent < len(self._sentences):
            return self._sentences[_sent]
        return ""


def resolve_thresholds(settings, provider_name: str) -> Tuple[float, float]:
    """Return (correct, partial) thresholds for a provider.

    Environment-level thresholds (CONCEPT_THRESHOLD_*) always win. Otherwise
    provider-specific defaults are used because TF-IDF cosine values sit lower
    than sentence-transformer values.
    """
    default_by_provider = {
        "tfidf": (0.50, 0.30),
        "sentence-transformers": (0.60, 0.45),
    }
    correct, partial = default_by_provider.get(provider_name, (0.60, 0.45))

    # Each override is applied independently. Requiring both to be set would
    # make CONCEPT_THRESHOLD_CORRECT alone look accepted in .env while having no
    # effect at all.
    if settings.concept_threshold_correct is not None:
        correct = float(settings.concept_threshold_correct)
    if settings.concept_threshold_partial is not None:
        partial = float(settings.concept_threshold_partial)

    if not (0.0 <= partial <= correct <= 1.0):
        raise ValueError(
            "Thresholds must satisfy 0 <= partial <= correct <= 1. "
            f"Got correct={correct}, partial={partial}."
        )
    return (correct, partial)