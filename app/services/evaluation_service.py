"""EvaluationService - orchestrates the whole pipeline:

route -> validate -> [OCR] -> concept extraction -> rubric score
      -> feedback -> recommendations -> persistence -> JSON response
"""

import time
from typing import List, Optional

from app.core.exceptions import EvaluationError, ResourceNotFoundError, ValidationError
from app.core.logging_config import get_logger
from app.extensions import db
from app.models import ConceptResult, Evaluation, LearningRecommendation, Question, Student
from app.services.concept_service import resolve_rubric
from app.services.feedback import generate_feedback
from app.services.recommendation_service import (
    build_recommendations,
    get_mastery_profile,
    update_mastery,
)
from app.serializers import serialize_evaluation
from app.services.rubric_service import RubricEngine
from sqlalchemy.exc import IntegrityError
from src.ml.inference import classify_answer, match_concepts
from src.ml.model import ModelManager, model_manager

logger = get_logger("services.evaluation")


class EvaluationService:
    """Stateless orchestrator; created per app via current_app settings."""

    def __init__(
        self,
        settings,
        manager: Optional[ModelManager] = None,
        ocr_service=None,
        rubric_engine: Optional[RubricEngine] = None,
        persist: bool = True,
    ) -> None:
        self.settings = settings
        self.manager = manager or model_manager
        self.ocr_service = ocr_service
        self.rubric_engine = rubric_engine or RubricEngine(
            partial_credit_factor=settings.partial_credit_factor,
            incorrect_penalty=settings.incorrect_penalty,
        )
        self.persist = persist

    # ------------------------------------------------------------------ #
    def run_text_evaluation(
        self,
        question: str,
        answer: str,
        max_score: int,
        rubric: Optional[dict] = None,
        student_id: Optional[int] = None,
        subject: Optional[str] = None,
        source: str = "text",
        ocr_meta: Optional[dict] = None,
    ) -> dict:
        """Full evaluation for a typed/pasted (or OCR'd) answer."""
        question = question.strip()
        answer = answer.strip()
        if not answer:
            raise ValidationError("Answer text is empty.")

        if not self.manager.loaded:
            self.manager.load(self.settings)

        t0 = time.perf_counter()
        try:
            resolved_rubric, rubric_source = resolve_rubric(question, rubric, max_score)
            concept_matches = match_concepts(question, answer, resolved_rubric["concepts"], self.manager)
            score_result = self.rubric_engine.score(concept_matches, resolved_rubric)
        except Exception as exc:
            logger.exception("Evaluation pipeline failed")
            raise EvaluationError("The evaluation pipeline could not produce a result.") from exc

        concept_results = score_result["concept_results"]
        score = score_result["score"]
        percentage = score_result["percentage"]

        # optional correctness prediction (None in development mode - honest).
        correctness = None
        try:
            correctness = classify_answer(question, answer, manager=self.manager)
        except Exception:  # noqa: BLE001
            logger.exception("Optional correctness classifier failed; continuing.")

        mastery = get_mastery_profile(student_id) if student_id is not None else []
        recommendations = build_recommendations(concept_results, mastery)
        feedback = generate_feedback(concept_results, score, max_score, percentage)

        evaluation = self._persist(
            question=question,
            answer=answer,
            source=source,
            student_id=student_id,
            subject=subject,
            max_score=max_score,
            score=score,
            percentage=percentage,
            feedback=feedback,
            rubric_source=rubric_source,
            concept_results=concept_results,
            recommendations=recommendations,
            resolved_rubric=resolved_rubric,
            ocr_meta=ocr_meta,
        )

        latency = time.perf_counter() - t0
        logger.info(
            "Evaluation completed | id=%s score=%s/%s mode=%s latency=%.2fs",
            evaluation.public_id if evaluation else "-",
            score, max_score, self.settings.model_mode, latency,
        )

        return serialize_evaluation(
            question=question,
            student_id=student_id,
            score=score,
            max_score=max_score,
            percentage=percentage,
            rubric_source=rubric_source,
            concept_results=concept_results,
            feedback=feedback,
            recommendations=recommendations,
            correctness=correctness,
            source=source,
            ocr_meta=ocr_meta,
            model_mode=str(self.settings.model_mode),
            evaluation_id=evaluation.public_id if evaluation else None,
            created_at=evaluation.created_at.isoformat() if evaluation else None,
        )

    # ------------------------------------------------------------------ #
    def evaluate_image(
        self,
        question: str,
        image_bytes: bytes,
        filename: str,
        max_score: int,
        rubric: Optional[dict] = None,
        student_id: Optional[int] = None,
        subject: Optional[str] = None,
    ) -> dict:
        """OCR the image, then run the standard evaluation."""
        if self.ocr_service is None:
            raise EvaluationError("OCR service is not configured for this app.")
        t0 = time.perf_counter()
        ocr_result = self.ocr_service.process(image_bytes, filename)
        logger.info(
            "OCR completed | engine=%s confidence=%s latency=%.2fs",
            ocr_result["engine"], ocr_result["confidence"], time.perf_counter() - t0,
        )
        return self.run_text_evaluation(
            question=question,
            answer=ocr_result["text"],
            max_score=max_score,
            rubric=rubric,
            student_id=student_id,
            subject=subject,
            source="image",
            ocr_meta={
                "engine": ocr_result["engine"],
                "confidence": ocr_result["confidence"],
                "pages": ocr_result["pages"],
            },
        )

    # ------------------------------------------------------------------ #
    def _persist(
        self,
        question: str,
        answer: str,
        source: str,
        student_id: Optional[int],
        subject: Optional[str],
        max_score: int,
        score: float,
        percentage: float,
        feedback: str,
        rubric_source: str,
        concept_results: List[dict],
        recommendations: List[dict],
        resolved_rubric: dict,
        ocr_meta: Optional[dict] = None,
    ):
        if not self.persist:
            return None

        # Reject a dangling student_id up front: SQLite would happily store an
        # orphan row while Postgres raises IntegrityError -> a 500 on write.
        if student_id is not None and db.session.get(Student, student_id) is None:
            raise ValidationError(
                f"Student {student_id} not found.",
                details={"student_id": student_id},
            )

        try:
            existing_q = Question.query.filter_by(question_text=question).first()
            if existing_q is None:
                existing_q = Question(
                    question_text=question,
                    subject=subject,
                    rubric_json=resolved_rubric,
                )
                db.session.add(existing_q)
            else:
                # Keep the newest rubric/subject rather than silently dropping them.
                existing_q.rubric_json = resolved_rubric
                if subject:
                    existing_q.subject = subject

            evaluation = Evaluation(
                student_id=student_id,
                answer_text=answer,
                source=source,
                score=score,
                max_score=max_score,
                percentage=percentage,
                feedback=feedback,
                rubric_source=rubric_source,
                model_mode=str(self.settings.model_mode),
                recommendations_json=[
                    {"topic": r["topic"], "kind": r["kind"], "text": r["text"]}
                    for r in recommendations
                ],
                ocr_json=ocr_meta,
            )
            db.session.add(evaluation)
            db.session.flush()
            evaluation.question_id = existing_q.id

            for row in concept_results:
                db.session.add(
                    ConceptResult(
                        evaluation_id=evaluation.id,
                        concept_name=row["concept"],
                        status=row["status"],
                        confidence=float(row.get("confidence", 0.0)),
                        evidence=row.get("evidence", ""),
                        weight=float(row.get("weight", 1.0)),
                    )
                )
            for rec in recommendations:
                db.session.add(
                    LearningRecommendation(
                        evaluation_id=evaluation.id,
                        topic=rec["topic"],
                        kind=rec["kind"],
                        text=rec["text"],
                    )
                )
            if student_id is not None:
                # Same transaction as the evaluation itself - a partial write
                # would otherwise be served as a complete evaluation.
                update_mastery(student_id, concept_results)

            db.session.commit()
        except Exception:
            db.session.rollback()
            raise
        return evaluation

    # ------------------------------------------------------------------ #
    def get_evaluation(self, evaluation_id: str) -> dict:
        evaluation = Evaluation.query.filter_by(public_id=evaluation_id).first()
        if evaluation is None:
            raise ResourceNotFoundError(f"Evaluation '{evaluation_id}' not found.")
        return evaluation.to_dict()

    def student_progress(self, student_id: int) -> dict:
        student = db.session.get(Student, student_id)
        if student is None:
            raise ResourceNotFoundError(f"Student {student_id} not found.")

        evaluations = (
            Evaluation.query.filter_by(student_id=student_id)
            .order_by(Evaluation.created_at.desc())
            .all()
        )
        mastery = get_mastery_profile(student_id)
        avg = sum(e.percentage for e in evaluations) / len(evaluations) if evaluations else 0.0

        strong = [m["concept"] for m in mastery if m["mastery"] >= 0.7]
        weak = [m["concept"] for m in mastery if m["mastery"] < 0.5]
        developing = [m["concept"] for m in mastery if 0.5 <= m["mastery"] < 0.7]

        recent = [
            {
                "evaluation_id": e.public_id,
                "question": (e.question.question_text if e.question else ""),
                "score": float(e.score),
                "max_score": e.max_score,
                "percentage": float(e.percentage),
                "created_at": e.created_at.isoformat(),
            }
            for e in evaluations[:10]
        ]
        return {
            "student": student.to_dict(),
            "average_score": round(avg, 1),
            "total_evaluations": len(evaluations),
            "strong_concepts": strong,
            "weak_concepts": weak,
            "developing_concepts": developing,
            "concept_mastery": mastery,
            "recent_evaluations": recent,
        }

    @staticmethod
    def get_or_create_student(name: str) -> tuple[Student, bool]:
        """Return `(student, created)`.

        `students.name` is unique, so a concurrent insert is caught and turned
        back into a plain lookup instead of an IntegrityError 500.
        """
        clean = name.strip()
        student = Student.query.filter_by(name=clean).first()
        if student is not None:
            return student, False
        student = Student(name=clean)
        db.session.add(student)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            existing = Student.query.filter_by(name=clean).first()
            if existing is None:
                raise
            return existing, False
        return student, True

    @staticmethod
    def list_students(limit: int = 50) -> List[dict]:
        rows = (
            Student.query.order_by(Student.created_at.desc(), Student.id.desc()).limit(limit).all()
        )
        return [row.to_dict() for row in rows]