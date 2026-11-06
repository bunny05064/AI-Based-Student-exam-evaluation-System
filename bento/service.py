"""BentoML serving layer for the evaluation model bundle.

Exposes the same scoring pipeline as the Flask app (concept extraction,
rubric scoring, feedback, recommendations) WITHOUT the web-server/DB layer:
a stateless, GPU-friendly service you can deploy with ``bentoml serve``.

Run locally:
    bentoml serve bento.service:EvaluationService

Build & ship:
    bentoml build -f bentofile.yaml
    bentoml containerize <bundle-name>:latest

Configuration comes from the process environment (see `.env.example`). With
nothing set it runs in development mode: TF-IDF embeddings + `NoOpClassifier`,
so `correctness_prediction` is absent rather than faked. Point `MODEL_MODE` at
`production` once a trained bundle exists in `models/`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import bentoml
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.services.concept_service import resolve_rubric
from app.services.feedback import generate_feedback
from app.services.recommendation_service import build_recommendations
from app.services.rubric_service import RubricEngine
from src.ml.inference import match_concepts
from src.ml.model import ModelManager, model_manager


class EvaluateTextInput(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    max_score: Optional[int] = Field(default=None, ge=1)
    rubric: Optional[dict] = None


class ConceptGroup(BaseModel):
    correct: List[str] = []
    partially_correct: List[str] = []
    missing: List[str] = []
    incorrect: List[str] = []
    results: List[Dict] = []


class EvaluateTextOutput(BaseModel):
    score: float
    max_score: int
    percentage: float
    rubric_source: str
    concepts: ConceptGroup
    feedback: str
    recommendations: List[Dict]
    metadata: Dict


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 120},
)
class EvaluationService:
    """Serves answer evaluation per request using the project model stack."""

    def __init__(self) -> None:
        # Read the real environment. The previous version passed a dict of
        # "defaults" to `Settings.from_env`, which (a) mutated `os.environ` for
        # the whole process and (b) forced MODEL_MODE=development even when an
        # operator had explicitly set production.
        self.settings = Settings.from_env()
        manager: ModelManager = model_manager
        manager.load(self.settings)
        self.manager = manager
        self.rubric_engine = RubricEngine(
            partial_credit_factor=self.settings.partial_credit_factor,
            incorrect_penalty=self.settings.incorrect_penalty,
        )

    # ------------------------------------------------------------------ #
    @bentoml.api
    def metadata(self) -> dict:
        correct, partial = self.manager.thresholds()
        return {
            "service": "ai-answer-evaluation",
            "model_mode": self.settings.model_mode,
            "embedding_provider": self.manager.provider_name,
            "classifier": self.manager.classifier_name,
            "concept_thresholds": {"correct": correct, "partial": partial},
            "rubric_engine": {
                "partial_credit_factor": self.settings.partial_credit_factor,
                "incorrect_penalty": self.settings.incorrect_penalty,
            },
        }

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok", "loaded": self.manager.loaded}

    # ------------------------------------------------------------------ #
    @bentoml.api
    def evaluate_text(self, payload: EvaluateTextInput) -> EvaluateTextOutput:
        question = payload.question.strip()
        answer = payload.answer.strip()
        if not answer:
            raise ValueError("'answer' must not be empty.")
        max_score = int(payload.max_score or self.settings.max_score_default)
        if max_score <= 0:
            raise ValueError("'max_score' must be a positive integer.")

        resolved_rubric, rubric_source = resolve_rubric(
            question, payload.rubric, max_score
        )
        concept_matches = match_concepts(
            question, answer, resolved_rubric["concepts"], self.manager
        )
        score_result = self.rubric_engine.score(concept_matches, resolved_rubric)

        concepts = score_result["concept_results"]
        score = score_result["score"]
        percentage = score_result["percentage"]

        return EvaluateTextOutput(
            score=float(score),
            max_score=max_score,
            percentage=float(percentage),
            rubric_source=rubric_source,
            concepts=ConceptGroup(
                results=concepts, **self._group_by_status(concepts)
            ),
            feedback=generate_feedback(concepts, score, max_score, percentage),
            recommendations=build_recommendations(concepts),
            metadata=self.metadata(),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_by_status(results: List[dict]) -> dict:
        groups: dict = {
            "correct": [],
            "partially_correct": [],
            "missing": [],
            "incorrect": [],
        }
        for row in results:
            groups.setdefault(row["status"], []).append(row["concept"])
        return groups