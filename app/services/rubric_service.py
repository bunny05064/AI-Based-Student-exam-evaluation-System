"""Deterministic, testable rubric scoring engine.

The score is a pure function of (concept statuses, weights, max_score) - the
model extracts concepts, the rubric engine converts them to a number. No LLM
is ever allowed to invent a score.
"""

import math
from typing import List

from app.core.exceptions import ValidationError
from src.data.rubrics import normalize_rubric

CORRECT = 1.0
STATUS_WEIGHTS = {"correct": 1.0, "partially_correct": None, "missing": 0.0, "incorrect": 0.0}


class RubricEngine:
    """Weighted concept-based scoring with deterministic rounding."""

    def __init__(self, partial_credit_factor: float = 0.5, incorrect_penalty: float = 0.0) -> None:
        self.partial_credit_factor = partial_credit_factor
        self.incorrect_penalty = incorrect_penalty

    # ------------------------------------------------------------------ #
    def score(self, concept_matches: List[dict], rubric: dict) -> dict:
        """Compute the final score from concept matches and a rubric.

        ``concept_matches``: list of {"concept", "status", "evidence",
        "confidence", "weight"}.
        ``rubric``: {"name", "max_score", "concepts": [{"name","weight",...}]}
        """
        try:
            rubric = normalize_rubric(rubric)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        matched_by_name = {m.get("concept"): m for m in concept_matches}

        rows: List[dict] = []
        total_weight = 0.0
        earned_weight = 0.0
        for concept in rubric["concepts"]:
            name = concept["name"]
            weight = concept["weight"]
            match = matched_by_name.get(name, {})
            status = str(match.get("status", "missing"))
            factor = self._status_factor(status)
            contribution = weight * factor
            rows.append(
                {
                    "concept": name,
                    "weight": weight,
                    "status": status,
                    "confidence": float(match.get("confidence", 0.0)),
                    "evidence": match.get("evidence", ""),
                    "factor": factor,
                    "contribution": round(contribution, 4),
                }
            )
            total_weight += weight
            earned_weight += contribution

        max_score = int(rubric["max_score"])
        if total_weight <= 0:
            raise ValidationError("Rubric has no concept weight.")

        raw = earned_weight / total_weight * max_score
        raw = max(0.0, min(float(max_score), raw))
        score = self._round_half_up(raw)
        percentage = round(score / max_score * 100) if max_score else 0

        return {
            "score": score,
            "max_score": max_score,
            "percentage": percentage,
            "method": "rubric-weighted",
            "total_weight": round(total_weight, 4),
            "earned_weight": round(earned_weight, 4),
            "concept_results": rows,
        }

    # ------------------------------------------------------------------ #
    def _status_factor(self, status: str) -> float:
        if status == "correct":
            return 1.0
        if status == "partially_correct":
            return self.partial_credit_factor
        if status == "incorrect":
            return -self.incorrect_penalty
        return 0.0  # missing / unknown

    @staticmethod
    def _round_half_up(value: float, base: float = 0.5) -> float:
        if base <= 0:
            raise ValueError("rounding base must be positive")
        return base * math.floor(float(value) / base + 0.5)