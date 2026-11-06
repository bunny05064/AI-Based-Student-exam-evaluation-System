"""Similarity helpers built on NumPy (no scipy dependency)."""

from typing import List, Tuple

import numpy as np


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(query_vectors: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix between two sets of rows."""
    q = l2_normalize(np.asarray(query_vectors, dtype=np.float64))
    c = l2_normalize(np.asarray(corpus_vectors, dtype=np.float64))
    return np.dot(q, c.T)


def most_similar(
    query_vector: np.ndarray, corpus_vectors: np.ndarray
) -> Tuple[int, float, float]:
    """Return (index, best_similarity) for query against a corpus."""
    sims = cosine_matrix(query_vector.reshape(1, -1), corpus_vectors)
    idx = int(np.argmax(sims[0]))
    best = float(sims[0][idx])
    second = float(np.sort(sims[0])[-2]) if sims.shape[1] > 1 else best
    return idx, best, second


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation; returns NaN-degenerate-safe values."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation via rank + Pearson."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    def ranks(values: List[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=float))
        ranks_out = np.empty_like(order, dtype=float)
        ranks_out[order] = np.arange(len(values))
        return ranks_out

    return pearson_correlation(list(ranks(x)), list(ranks(y)))