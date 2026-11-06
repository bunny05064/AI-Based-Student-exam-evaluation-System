"""Built-in rubric registry and automatic rubric generation.

Rubrics drive the explainable scoring. They are plain JSON/dict structures:
    {"concepts": [{"name", "weight", "synonyms": [...],
                   "incorrect_phrases": [...]}], "max_score": int}

- ``find_builtin_rubric``: keyword-matched curated rubrics for common demo
  school questions (honest, small, documented set).
- ``auto_rubric_from_question``: heuristic keyphrase-derived concepts with
  equal weights (clearly labelled ``auto``).
"""

import re
from typing import List, Optional

from app.utils.text import clean_text

BUILTIN_RUBRICS: List[dict] = [
    {
        "name": "Newton's Second Law of Motion",
        "keys": ["newton", "second law", "force and acceleration", "f=ma"],
        "max_score": 10,
        "concepts": [
            {
                "name": "force",
                "weight": 1,
                "synonyms": ["net force", "an external push or pull", "f"],
            },
            {
                "name": "mass",
                "weight": 1,
                "synonyms": ["inertia", "amount of matter", "m"],
            },
            {
                "name": "acceleration",
                "weight": 1,
                "synonyms": [
                    "rate of change of velocity",
                    "change in velocity",
                    "change of velocity",
                    "velocity change",
                    "speeds up or slows down",
                    "a",
                ],
            },
            {
                "name": "F = ma",
                "weight": 2,
                "synonyms": [
                    "force equals mass times acceleration",
                    "force equals mass multiplied by acceleration",
                    "force depend on mass and acceleration",
                    "f = m a",
                    "force is proportional to mass and acceleration",
                ],
            },
        ],
    },
    {
        "name": "Photosynthesis",
        "keys": ["photosynthesis", "food making process in plants", "green plants make food"],
        "max_score": 10,
        "concepts": [
            {
                "name": "sunlight",
                "weight": 1,
                "synonyms": ["light energy", "solar energy", "sun", "light"],
            },
            {
                "name": "chlorophyll",
                "weight": 1,
                "synonyms": ["green pigment", "photosynthetic pigment", "chloroplast"],
            },
            {
                "name": "carbon dioxide",
                "weight": 2,
                "synonyms": ["co2", "carbon-di-oxide", "co 2"],
            },
            {
                "name": "water",
                "weight": 1,
                "synonyms": ["h2o", "moisture"],
            },
            {
                "name": "glucose",
                "weight": 2,
                "synonyms": ["sugar", "food", "carbohydrate", "starch", "carbohydrates"],
            },
            {
                "name": "oxygen",
                "weight": 1,
                "synonyms": ["o2", "releases oxygen"],
            },
        ],
    },
    {
        "name": "Water Cycle",
        "keys": ["water cycle", "hydrologic cycle", "water recycle"],
        "max_score": 10,
        "concepts": [
            {
                "name": "evaporation",
                "weight": 2,
                "synonyms": ["water turns to vapour", "vapour", "liquid to gas"],
            },
            {
                "name": "condensation",
                "weight": 2,
                "synonyms": ["clouds form", "vapour becomes droplets", "gas to liquid"],
            },
            {
                "name": "precipitation",
                "weight": 2,
                "synonyms": ["rain", "rainfall", "snow", "water falls back"],
            },
            {
                "name": "storage",
                "weight": 1,
                "synonyms": ["lakes", "oceans", "groundwater", "collection"],
            },
        ],
    },
]

DEFAULT_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for with from by is are was were be been
    being explain state describe what why how does do did it its this that these
    those please briefly detail short note answer write about
    """.split()
)

# Concept words that merely look like plurals. The previous filter was
# `not t.endswith("s")`, which silently discarded mass, gas, process, loss,
# class, physics and every other content word ending in "s".
IRREGULAR_PLURALS = frozenset(
    """
    gas gases mass masses process processes physics kinetics mechanics osmosis
    analysis hypothesis emphasis synthesis hydrolysis photosynthesis metabolism
    class classes loss losses stress stress
    """.split()
)


def _is_plural_noise(token: str) -> bool:
    """True for a token that is only a weak plural of another concept."""
    if token in IRREGULAR_PLURALS:
        return False
    return token.endswith("s") and not token.endswith("ss") and len(token) > 3


def find_builtin_rubric(question: str) -> Optional[dict]:
    """Return a curated rubric if the question matches one, else None."""
    cleaned = clean_text(question).lower()
    for rubric in BUILTIN_RUBRICS:
        words = cleaned.split()
        for key in rubric["keys"]:
            if key.strip() in cleaned or key in words:
                return rubric
    return None


def auto_rubric_from_question(question: str, max_score: int = 10) -> dict:
    """Heuristic rubric: key concepts extracted from the question itself.

    This is clearly a fallback - concepts are the question's own content words
    (with equal weights), so 'missing concept' detection is only meaningful
    when the student omits part of what the question asks for.
    """
    cleaned = clean_text(question).lower()
    tokens = re.findall(r"[a-z][a-z\_\-]{2,}", cleaned)
    meaningful = [
        t
        for t in tokens
        if t not in DEFAULT_STOPWORDS
        and len(t) > 3
        and not _is_plural_noise(t)
    ]

    # Prefer compounds first, then single words.
    ordered: List[str] = []
    for left, right in zip(meaningful, meaningful[1:]):
        ordered.append(f"{left} {right}")
    for token in meaningful:
        if token not in flatten(ordered):
            ordered.append(token)

    concepts = []
    seen = set()
    for phrase in ordered[:8]:
        if phrase in seen:
            continue
        seen.add(phrase)
        concepts.append({"name": phrase, "weight": 1, "synonyms": []})

    if not concepts:
        concepts = [{"name": "concept", "weight": 1, "synonyms": []}]
    return {"name": "Auto-generated", "max_score": max_score, "concepts": concepts}


def flatten(phrases: List[str]) -> List[str]:
    return [w for p in phrases for w in p.split()]


def normalize_rubric(rubric: dict) -> dict:
    """Validate and normalize a rubric input dict."""
    concepts = rubric.get("concepts")
    if not isinstance(concepts, list) or len(concepts) == 0:
        raise ValueError("Rubric requires a non-empty 'concepts' list.")
    max_score = int(rubric.get("max_score", 10))
    if max_score <= 0:
        raise ValueError("Rubric max_score must be positive.")

    cleaned: List[dict] = []
    for concept in concepts:
        name = str(concept.get("name", "")).strip()
        if not name:
            continue
        cleaned.append(
            {
                "name": name,
                "weight": float(concept.get("weight", 1.0)),
                "synonyms": [str(s) for s in (concept.get("synonyms") or []) if str(s).strip()],
                "incorrect_phrases": [
                    str(s) for s in (concept.get("incorrect_phrases") or []) if str(s).strip()
                ],
            }
        )
    if not cleaned:
        raise ValueError("Rubric concepts are all empty.")
    return {"name": rubric.get("name", "Rubric"), "max_score": max_score, "concepts": cleaned}