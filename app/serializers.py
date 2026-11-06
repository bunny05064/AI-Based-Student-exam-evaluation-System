"""Canonical JSON shapes for API responses.

Lives outside `app.models` and `app.services` so that the ORM layer and the
service layer can both depend on it without an import cycle.
"""

from typing import List, Optional

from app.services.feedback import improvement_tip

CONCEPT_STATUSES = ("correct", "partially_correct", "missing", "incorrect")


def serialize_evaluation(
    *,
    question: Optional[str],
    student_id: Optional[int],
    score: float,
    max_score: int,
    percentage: float,
    rubric_source: str,
    concept_results: List[dict],
    feedback: str,
    recommendations: List[dict],
    correctness: Optional[dict] = None,
    source: str = "text",
    ocr_meta: Optional[dict] = None,
    model_mode: str = "development",
    evaluation_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict:
    """The single response shape for an evaluation.

    Both the live `POST /evaluate` response and the stored
    `GET /evaluations/<id>` response are built here, so they cannot drift. The
    result page used to read `rubric_source` and `ocr` from a stored
    evaluation that never carried them, and rendered "Rubric source: ." after
    a page refresh.
    """

    def by_status(status: str) -> List[str]:
        return [c["concept"] for c in concept_results if c["status"] == status]

    payload = {
        "evaluation_id": evaluation_id,
        "student_id": student_id,
        "question": question,
        "score": float(score),
        "max_score": max_score,
        "percentage": float(percentage),
        "concepts": {
            "correct": by_status("correct"),
            "partially_correct": by_status("partially_correct"),
            "missing": by_status("missing"),
            "incorrect": by_status("incorrect"),
            "results": concept_results,
        },
        "feedback": feedback,
        "improvement_tip": improvement_tip(concept_results),
        "recommendations": [dict(r) for r in recommendations],
        "rubric_source": rubric_source,
        "correctness_prediction": correctness,
        "model_mode": str(model_mode),
        "source": source,
        "ocr": ocr_meta,
    }
    if created_at is not None:
        payload["created_at"] = created_at
    return payload