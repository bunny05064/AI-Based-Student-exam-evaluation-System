"""Text utilities: deterministic, dependency-light NLP helpers."""

import re
from typing import List

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Mathematical tokens that we intentionally preserve.
_MATH_PATTERN = (
    r"((?:[A-Za-z]+)=[A-Za-z0-9*/+\-^_()\s.]+)|(?:[A-Za-z0-9_]+(?:\\frac|\^)[^\s]+)"
)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into single spaces and strip edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """Unicode NFKC normalization (keeps math symbols, fixes strange quotes)."""
    import unicodedata

    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """Full normalization pipeline used before embedding/similarity."""
    if text is None:
        return ""
    text = normalize_unicode(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_empty(text: str) -> bool:
    return not bool(text) or not clean_text(text).strip()


def segment_sentences(text: str) -> List[str]:
    """Split text into sentences on sentence-ending punctuation.

    Abbreviations (e.g. "U.S.") are not handled; acceptable for evaluation
    evidence extraction.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def truncate(text: str, max_chars: int) -> str:
    """Hard-truncate keeping the last 24 chars of the truncated suffix for context."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30].rstrip() + "... [truncated by system]"


_NEGATIONS = (
    " not ",
    "n't",
    " never ",
    " no ",
    " cannot ",
    " can't ",
    " isn't ",
    " doesn't ",
    " isn't relevant",
    " irrelevant",
    " unrelated",
    " incorrect",
)


def contains_negation(sentence: str) -> bool:
    """Heuristic: does the sentence express a negative/denial near a key term?"""
    lowered = f" {sentence.lower()} "
    return any(neg in lowered for neg in _NEGATIONS)


def remove_stopwords_light(tokens: List[str], stopwords) -> List[str]:
    """Light stopword removal (keeps contextual words for transformers)."""
    return [t for t in tokens if t.lower() not in stopwords]


def preserve_math_terms(text: str) -> str:
    """Return math-like substrings found in text (never drops surrounding text)."""
    return " ".join(re.findall(_MATH_PATTERN, text)).strip()


def jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)