from app.extensions import db
from app.utils import utcnow


class Question(db.Model):
    """A question plus the rubric that was resolved for it.

    `rubric_json` stores the rubric actually used to score the answer
    (`{"name": ..., "concepts": [...]}`). Persisting it is what makes a stored
    score auditable: without it, a score cannot be re-derived or explained
    after the built-in catalogue changes.
    """

    __tablename__ = "questions"
    __table_args__ = (
        db.UniqueConstraint("question_text", name="uq_questions_question_text"),
    )

    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.Text, nullable=False, index=True)
    subject = db.Column(db.String(120), index=True, nullable=True)
    rubric_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    evaluations = db.relationship("Evaluation", back_populates="question", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question_text,
            "subject": self.subject,
            "rubric": self.rubric_json,
            "created_at": self.created_at.isoformat(),
        }