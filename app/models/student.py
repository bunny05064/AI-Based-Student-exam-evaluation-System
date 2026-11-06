from app.extensions import db
from app.utils import utcnow


class Student(db.Model):
    __tablename__ = "students"
    __table_args__ = (db.UniqueConstraint("name", name="uq_students_name"),)

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    evaluations = db.relationship(
        "Evaluation",
        back_populates="student",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )
    mastery_levels = db.relationship(
        "ConceptMastery",
        back_populates="student",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "created_at": self.created_at.isoformat()}