from app.extensions import db
from app.utils import utcnow

VALID_STATUSES = ("correct", "partially_correct", "missing", "incorrect")
VALID_KINDS = ("topic", "question")


class ConceptResult(db.Model):
    __tablename__ = "concept_results"
    __table_args__ = (
        db.CheckConstraint(
            "status IN ('correct', 'partially_correct', 'missing', 'incorrect')",
            name="ck_concept_results_status_valid",
        ),
        db.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_concept_results_confidence_range"
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    evaluation_id = db.Column(
        db.Integer,
        db.ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    concept_name = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(24), nullable=False)  # correct|partially_correct|missing|incorrect
    confidence = db.Column(db.Float, nullable=False)
    evidence = db.Column(db.Text, nullable=True)
    weight = db.Column(db.Float, nullable=False, default=1.0)

    evaluation = db.relationship("Evaluation", back_populates="concept_results")

    def to_dict(self) -> dict:
        return {
            "concept": self.concept_name,
            "status": self.status,
            "evidence": self.evidence or "",
            "confidence": round(float(self.confidence), 4),
            "weight": float(self.weight),
        }


class ConceptMastery(db.Model):
    """Exponentially moving mastery estimate per (student, concept)."""

    __tablename__ = "concept_mastery"
    __table_args__ = (
        db.UniqueConstraint("student_id", "concept_name", name="uq_student_concept"),
        db.CheckConstraint(
            "mastery >= 0 AND mastery <= 1", name="ck_concept_mastery_range"
        ),
        db.CheckConstraint("attempts >= 0", name="ck_concept_mastery_attempts_non_negative"),
    )

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.Integer,
        db.ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    concept_name = db.Column(db.String(200), nullable=False, index=True)
    mastery = db.Column(db.Float, nullable=False, default=0.0)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    student = db.relationship("Student", back_populates="mastery_levels")

    def to_dict(self) -> dict:
        return {
            "concept": self.concept_name,
            "mastery": round(float(self.mastery), 4),
            "attempts": self.attempts,
        }


class LearningRecommendation(db.Model):
    __tablename__ = "learning_recommendations"
    __table_args__ = (
        db.CheckConstraint(
            "kind IN ('topic', 'question')", name="ck_learning_recommendations_kind_valid"
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    evaluation_id = db.Column(
        db.Integer,
        db.ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic = db.Column(db.String(200), nullable=False)
    kind = db.Column(db.String(24), nullable=False)  # topic | question
    text = db.Column(db.Text, nullable=False)

    evaluation = db.relationship("Evaluation", back_populates="recommendations")

    def to_dict(self) -> dict:
        return {"topic": self.topic, "kind": self.kind, "text": self.text}