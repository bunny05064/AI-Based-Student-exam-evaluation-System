"""Embedding providers.

Two providers share one interface (``encode_texts`` -> matrix):

- ``TfIdfEmbeddingProvider``: pure classical baseline using scikit-learn.
  Runs without torch/transformers; used in development mode and for
  Experiment 1 (TF-IDF + classical ML).
- ``SentenceTransformerEmbeddingProvider``: real transformer embeddings
  (requires `requirements-ml.txt`).

Both are real, deterministic similarity signals - a TF-IDF fallback is NOT a
mock prediction.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.core.exceptions import ModelLoadError
from src.ml.similarity import cosine_matrix

# Token pattern keeps letters, digits and common math symbols together
# (e.g. "F=ma", "v^2", "Δ") instead of splitting them apart.
_TOKEN_PATTERN = r"(?u)\b[\w.+^=\-/*']+\b"


class EmbeddingProvider(ABC):
    """Interface for any component that maps strings to vectors."""

    name: str = "base"

    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into a (n, d) float32 array.

        The vector space is shared across the whole call; the caller must
        pass ALL texts it wants to compare in one invocation.
        """

    def encode(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def similarity_matrix(
        self, query_texts: List[str], corpus_texts: List[str]
    ) -> np.ndarray:
        """Pairwise (nq, nc) similarity between queries and corpus.

        Default: cosine similarity in the shared embedding space (this is
        correct for transformer providers). Providers that use a different
        classical measure override this method.
        """
        combined = list(query_texts) + list(corpus_texts)
        matrix = self.encode_texts(combined)
        queries = matrix[: len(query_texts)]
        corpus = matrix[len(query_texts):]
        return cosine_matrix(queries, corpus)


class TfIdfEmbeddingProvider(EmbeddingProvider):
    """Classical TF-IDF baseline embeddings (scikit-learn, no torch).

    ``encode_texts`` keeps L2-normalized TF-IDF rows (vector space, usable for
    training features). For concept matching we override ``similarity_matrix``
    with a corpus-IDF weighted character-shingle Jaccard - a standard classical
    short-text similarity that is much more discriminative than raw TF-IDF
    cosine on short concept phrases vs long answers.
    """

    name = "tfidf"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._vectorizer = None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=_TOKEN_PATTERN,
        )
        matrix = self._vectorizer.fit_transform(texts)
        return np.asarray(matrix.toarray(), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def similarity_matrix(
        self, query_texts: List[str], corpus_texts: List[str]
    ) -> np.ndarray:
        return weighted_shingle_coverage(query_texts, corpus_texts, k=4)


def _character_shingles(text: str, k: int) -> set:
    """k-character shingles preserving spaces (captures phrase boundaries)."""
    if not isinstance(text, str) or not text.strip():
        return set()
    compact = " ".join(text.lower().split())
    if len(compact) < k:
        return {compact} if compact else set()
    return set(compact[i : i + k] for i in range(len(compact) - k + 1))


def weighted_shingle_coverage(
    query_texts: List[str], corpus_texts: List[str], k: int = 4
) -> np.ndarray:
    """How much of each query (concept) phrase is covered by each corpus text.

    score(q, c) = sum_{s in shingles(q) and s in shingles(c)} idf(s)
                  / sum_{s in shingles(q)} idf(s)

    A concept phrase that literally appears in the answer gets ~1.0; one that
    is completely absent gets ~0.0; partial co-occurrence gives partial credit.
    This is the "concept presence/absence" measure for the classical baseline
    and is intentionally asymmetric (a long answer does not dilute the score).
    """
    from collections import Counter
    import math

    if not query_texts or not corpus_texts:
        return np.zeros((len(query_texts), len(corpus_texts)), dtype=np.float64)

    query_sets = [_character_shingles(t, k) for t in query_texts]
    corpus_sets = [_character_shingles(t, k) for t in corpus_texts]

    counts: Counter = Counter()
    for sset in corpus_sets:
        counts.update(sset)
    n_corpus = len(corpus_sets)

    def idf(shingle: str) -> float:
        count = counts.get(shingle, 0)
        return math.log((n_corpus + 1) / (count + 1)) + 1.0

    result = np.zeros((len(query_sets), len(corpus_sets)), dtype=np.float64)
    for qi, qset in enumerate(query_sets):
        if not qset:
            continue
        q_weight = sum(idf(s) for s in qset)
        if q_weight <= 0:
            continue
        for ci, cset in enumerate(corpus_sets):
            overlap = qset & cset
            if overlap:
                result[qi, ci] = sum(idf(s) for s in overlap) / q_weight
    return result


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Transformer embedding provider (sentence-transformers under the hood)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_dependency(self) -> None:
        """Eager import-only check (allows early fallback to the classical
        provider when sentence-transformers is not installed)."""
        from importlib.util import find_spec

        if find_spec("sentence_transformers") is None:
            raise ModelLoadError(
                "Sentence-transformers is not installed. Install "
                "requirements-ml.txt, or set EMBEDDING_FALLBACK=tfidf to use "
                "the classical baseline."
            )

    def _load(self) -> None:
        self._ensure_dependency()
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001 - surfaced as a safe error
            raise ModelLoadError(
                f"Failed to load embedding model '{self.model_name}'."
            ) from exc

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            ),
            dtype=np.float32,
        )


_CLASSICAL_NAMES = {"tfidf", "classical", "development", "auto"}


def resolve_embedding_provider(settings) -> EmbeddingProvider:
    """Pick an embedding provider based on configuration and availability."""
    requested = (settings.embedding_model or "").strip().lower()

    use_classical = (
        requested in _CLASSICAL_NAMES or settings.model_mode == "development"
    )

    if use_classical:
        return TfIdfEmbeddingProvider(seed=settings.test_seed)

    try:
        provider = SentenceTransformerEmbeddingProvider(settings.embedding_model)
        provider._ensure_dependency()
        return provider
    except ModelLoadError:
        if settings.embedding_fallback == "tfidf":
            return TfIdfEmbeddingProvider(seed=settings.test_seed)
        raise


def provider_display_name(provider: EmbeddingProvider) -> str:
    return provider.name