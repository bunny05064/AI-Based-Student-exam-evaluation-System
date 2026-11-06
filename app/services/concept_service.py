"""Resolution of expected concepts for a question.

Order of preference:
1. rubric supplied in the API request  -> source = "provided"
2. a curated built-in rubric          -> source = "builtin"
3. heuristic keyphrases from the question -> source = "auto"
"""

from typing import Tuple

from src.data.rubrics import (
    auto_rubric_from_question,
    find_builtin_rubric,
    normalize_rubric,
)


def resolve_rubric(
    question: str, optional_rubric: dict | None, max_score: int
) -> Tuple[dict, str]:
    """Return (rubric_dict, source). max_score from the request wins."""
    if optional_rubric:
        rubric = normalize_rubric(optional_rubric)
        source = "provided"
    else:
        builtin = find_builtin_rubric(question)
        if builtin is not None:
            rubric = normalize_rubric(builtin)
            source = "builtin"
        else:
            rubric = normalize_rubric(auto_rubric_from_question(question, max_score))
            source = "auto"

    rubric["max_score"] = max_score
    return rubric, source