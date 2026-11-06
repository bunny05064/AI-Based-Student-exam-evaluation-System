from uuid import uuid4

from app.extensions import db
from app.serializers import serialize_evaluation
from app.utils import utcnow

VALID_SOURCES = ("text", "image")
VALID_RUBRIC_SOURCES = ("provided", "auto", "builtin")


class Evaluation(db.Model):
    __tablename__ = "evaluations"
    __table_args__ = (
        db.CheckConstraint("score >= 0", name="ck_evaluations_score_non_negative"),
        db.CheckConstraint("max_score > 0", name="ck_evaluations_max_score_positive"),
        db.CheckConstraint("score <= max_score", name="ck_evaluations_score_within_max"),
        db.CheckConstraint(
            "percentage >= 0 AND percentage <= 100", name="ck_evaluations_percentage_range"
        ),
        db.CheckConstraint(
            "source IN ('text', 'image')", name="ck_evaluations_source_valid"
        ),
        db.CheckConstraint(
            "rubric_source IN ('provided', 'auto', 'builtin')",
            name="ck_evaluations_rubric_source_valid",
        ),
        # Serves the dashboard query: WHERE student_id = ? ORDER BY created_at DESC
        db.Index("ix_evaluations_student_created", "student_id", "created_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(32), unique=True, index=True, default=lambda: uuid4().hex[:16])
    student_id = db.Column(
        db.Integer, db.ForeignKey("students.id", ondelete="SET NULL"), nullable=True, index=True
    )
    question_id = db.Column(
        db.Integer, db.ForeignKey("questions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    answer_text = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(16), nullable=False, default="text")  # text | image
    score = db.Column(db.Float, nullable=False)
    max_score = db.Column(db.Integer, nullable=False)
    percentage = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    rubric_source = db.Column(db.String(32), nullable=False, default="auto")  # provided|auto|builtin
    model_mode = db.Column(db.String(16), nullable=False, default="development")
    recommendations_json = db.Column(db.JSON, nullable=True)
    ocr_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False, index=True)

    student = db.relationship("Student", back_populates="evaluations")
    question = db.relationship("Question", back_populates="evaluations")
    concept_results = db.relationship(
        "ConceptResult",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )
    recommendations = db.relationship(
        "LearningRecommendation",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    def to_dict(self) -> dict:
        """Stored-evaluation response, via the shared serializer.

        Must match the live `POST /evaluate` shape exactly - see
        `app.serializers.serialize_evaluation`.
        """
        return serialize_evaluation(
            question=self.question.question_text if self.question else None,
            student_id=self.student_id,
            score=float(self.score),
            max_score=self.max_score,
            percentage=float(self.percentage),
            rubric_source=self.rubric_source,
            concept_results=[c.to_dict() for c in self.concept_results],
            feedback=self.feedback,
            recommendations=self.recommendations_json or [],
            correctness=None,  # not stored: predictions are never persisted
            source=self.source,
            ocr_meta=self.ocr_json,
            model_mode=self.model_mode,
            evaluation_id=self.public_id,
            created_at=self.created_at.isoformat(),
        )


__all__ = ["Evaluation", "VALID_RUBRIC_SOURCES", "VALID_SOURCES"]

