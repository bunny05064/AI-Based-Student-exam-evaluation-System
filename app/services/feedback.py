"""Grounded, rule-based feedback generation.

Every sentence is derived from the concept results + score. No LLM, so
feedback can never contradict the computed score. Evidence quotes ground each
claim.
"""

from typing import List


def _quote(evidence: str) -> str:
    evidence = evidence.strip()
    if not evidence:
        return ""
    evidence = evidence if evidence.endswith((".", "!", "?")) else evidence + "."
    return f'"{evidence}"'


def summarize(score: float, max_score: int, percentage: float) -> str:
    if percentage >= 80:
        grade = "excellent"
    elif percentage >= 60:
        grade = "good"
    elif percentage >= 40:
        grade = "satisfactory"
    else:
        grade = "weak"
    return (
        f"Overall score: {score:g}/{max_score:g} ({percentage}%). "
        f"The answer is rated {grade} by the rubric engine."
    )


def generate_feedback(concept_results: List[dict], score: float, max_score: int, percentage: float) -> str:
    """Rule-based feedback fully grounded in per-concept evidence."""
    lines: List[str] = [summarize(score, max_score, percentage)]

    correct = [c for c in concept_results if c["status"] == "correct"]
    partial = [c for c in concept_results if c["status"] == "partially_correct"]
    missing = [c for c in concept_results if c["status"] == "missing"]
    incorrect = [c for c in concept_results if c["status"] == "incorrect"]

    if correct:
        names = ", ".join(f"'{c['concept']}'" for c in correct)
        lines.append(f"Your answer correctly covers: {names}.")
        for c in correct[:3]:
            quote = _quote(c.get("evidence", ""))
            if quote:
                lines.append(f"  - {c['concept']}: evident in {quote}")
    else:
        lines.append("No expected concept was fully captured in the answer.")

    if partial:
        names = ", ".join(f"'{c['concept']}'" for c in partial)
        lines.append(
            f"The concepts {names} are mentioned but not expressed clearly enough; "
            "state them precisely to earn full credit."
        )

    if incorrect:
        names = ", ".join(f"'{c['concept']}'" for c in incorrect)
        lines.append(
            f"The answer introduces statements that contradict the expected "
            f"concepts {names} - revise these points."
        )

    if missing:
        names = ", ".join(f"'{c['concept']}'" for c in missing)
        lines.append(f"To improve, explicitly include the missing concept(s): {names}.")
        first = missing[0]
        lines.append(
            f"Suggested focus: add a clear statement about '{first['concept']}' "
            "with a short correct explanation."
        )

    return "\n".join(lines)


def improvement_tip(concept_results: List[dict]) -> str:
    """One-line actionable tip pointing at the weakest gap."""
    missing = [c for c in concept_results if c["status"] == "missing"]
    incorrect = [c for c in concept_results if c["status"] == "incorrect"]
    if incorrect:
        return f"Correct the misunderstandings around: {', '.join(c['concept'] for c in incorrect[:2])}."
    if missing:
        return f"Review and add: {', '.join(c['concept'] for c in missing[:3])}."
    partial = [c for c in concept_results if c["status"] == "partially_correct"]
    if partial:
        return "Make the partially-covered concepts more explicit and precise."
    return "Solid answer - try extending with more detail and examples."