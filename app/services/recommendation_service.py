"""Personalized learning: a simple concept-mastery model + recommendations.

Mastery model: exponentially weighted moving average per (student, concept).
    new_mastery = 0.7 * old_mastery + 0.3 * attempt_value
attempt_value: correct=1.0, partially_correct=0.5, missing=0.0, incorrect=0.0

This is deliberately simple; the schema is structured so a Knowledge Tracing
model can be dropped in later (ConceptMastery table + attempt history).
"""

from typing import List, Sequence

from app.models import ConceptMastery, db


def _attempt_value(status: str) -> float:
    return {"correct": 1.0, "partially_correct": 0.5, "missing": 0.0, "incorrect": 0.0}.get(
        status, 0.0
    )


def update_mastery(student_id: int, concept_results: Sequence[dict]) -> None:
    """Upsert EMA mastery for every concept in the latest evaluation.

    Does NOT commit: the caller owns the transaction so that an evaluation and
    its mastery rows land atomically.
    """
    from app.utils import utcnow

    for result in concept_results:
        concept = str(result.get("concept", "")).strip()
        if not concept:
            continue
        value = _attempt_value(result.get("status", "missing"))
        row = (
            ConceptMastery.query.filter_by(student_id=student_id, concept_name=concept).first()
        )
        if row is None:
            row = ConceptMastery(
                student_id=student_id, concept_name=concept, mastery=value, attempts=1
            )
            db.session.add(row)
        else:
            row.mastery = 0.7 * row.mastery + 0.3 * value
            row.attempts += 1
            row.updated_at = utcnow()


def get_mastery_profile(student_id: int) -> List[dict]:
    rows = ConceptMastery.query.filter_by(student_id=student_id).order_by(
        ConceptMastery.mastery.asc()
    )
    return [row.to_dict() for row in rows]


def strength_of(mastery: float) -> str:
    if mastery >= 0.7:
        return "strong"
    if mastery < 0.5:
        return "weak"
    return "developing"


_recommendation_templates = {
    "topic": "Revise '{concept}': consult your textbook chapter on {concept} and read a short summary before attempting questions.",
    "conceptual_question": "Attempt 2 conceptual questions on '{concept}' that ask you to *explain* the idea in your own words.",
    "numerical_question": "Attempt 3 basic numerical questions on '{concept}' and recheck each step against the expected concepts.",
    "incorrect_fix": "Re-read the correct definition of '{concept}' and write one corrected sentence explaining it.",
}


def build_recommendations(concept_results: Sequence[dict], mastery: List[dict] = None) -> List[dict]:
    """Deterministic recommendation list from concept gaps + mastery profile."""
    recommendations: List[dict] = []
    weak_concepts: List[dict] = []

    for result in concept_results or []:
        concept = str(result.get("concept", "")).strip()
        status = result.get("status", "missing")
        if not concept:
            continue
        if status in ("missing", "incorrect"):
            recommendations.append(
                {
                    "topic": concept,
                    "kind": "topic",
                    "text": _recommendation_templates["topic"].format(concept=concept),
                }
            )
            recommendations.append(
                {
                    "topic": concept,
                    "kind": "question",
                    "text": _recommendation_templates["conceptual_question"].format(concept=concept),
                }
            )
            if status == "incorrect":
                recommendations.append(
                    {
                        "topic": concept,
                        "kind": "question",
                        "text": _recommendation_templates["incorrect_fix"].format(concept=concept),
                    }
                )
        if status != "correct":
            recommendations.append(
                {
                    "topic": concept,
                    "kind": "question",
                    "text": _recommendation_templates["numerical_question"].format(concept=concept),
                }
            )

    for profile in mastery or []:
        if strength_of(profile["mastery"]) == "weak":
            weak_concepts.append(profile)

    for profile in weak_concepts:
        concept = profile["concept"]
        already = any(r["kind"] == "topic" and r["topic"] == concept for r in recommendations)
        if not already:
            recommendations.append(
                {
                    "topic": concept,
                    "kind": "topic",
                    "text": (
                        f"Weak concept detected: '{concept}' (mastery ≈ "
                        f"{round(profile['mastery'] * 100):d}%). {_recommendation_templates['topic'].format(concept=concept)}"
                    ),
                }
            )
    return recommendations