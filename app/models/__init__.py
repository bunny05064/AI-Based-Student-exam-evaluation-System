"""SQLAlchemy models.

Relationship design (one -> many unless noted):
Student -> Evaluation -> ConceptResult
Student -> ConceptMastery
Question -> Evaluation   (the resolved rubric lives on Question.rubric_json)
Evaluation -> LearningRecommendation
"""

from app.extensions import db

from .concept import ConceptMastery, ConceptResult, LearningRecommendation
from .evaluation import Evaluation
from .question import Question
from .student import Student

__all__ = [
    "db",
    "Student",
    "Question",
    "Evaluation",
    "ConceptResult",
    "ConceptMastery",
    "LearningRecommendation",
]