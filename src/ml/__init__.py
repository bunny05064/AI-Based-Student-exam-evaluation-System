"""Machine learning pipeline package.

- embeddings: TF-IDF (classical) and sentence-transformer providers
- tokenizer: HF + simple fallback
- concept_extractor: semantic concept -> status matching (explainability core)
- classifier: optional correctness classifiers (dev = NoOp)
- similarity / evaluation: metric utilities
- model: ModelManager process-lifetime singleton
- inference / training: high-level wiring
"""

from src.ml.classifier import (
    ClassificationResult,
    CorrectnessClassifier,
    MLBaselineClassifier,
    NoOpClassifier,
    TransformerClassifier,
    load_classifier,
)
from src.ml.concept_extractor import (
    VALID_STATUSES,
    ConceptExtractor,
    ConceptMatch,
    resolve_thresholds,
)
from src.ml.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    TfIdfEmbeddingProvider,
    resolve_embedding_provider,
)
from src.ml.model import ModelManager, model_manager
from src.ml.similarity import (
    cosine_matrix,
    cosine_similarity,
    l2_normalize,
    most_similar,
)
from src.ml.tokenizer import HFTokenizer, SimpleTokenizer, load_tokenizer

__all__ = [
    "ClassificationResult",
    "CorrectnessClassifier",
    "MLBaselineClassifier",
    "NoOpClassifier",
    "TransformerClassifier",
    "load_classifier",
    "VALID_STATUSES",
    "ConceptExtractor",
    "ConceptMatch",
    "resolve_thresholds",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "TfIdfEmbeddingProvider",
    "resolve_embedding_provider",
    "ModelManager",
    "model_manager",
    "cosine_matrix",
    "cosine_similarity",
    "l2_normalize",
    "most_similar",
    "HFTokenizer",
    "SimpleTokenizer",
    "load_tokenizer",
]