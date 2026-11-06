"""Evaluation endpoints: /api/v1/evaluate and evaluation lookup."""

from flask import Blueprint, current_app, jsonify

from app.core.exceptions import ResourceNotFoundError
from app.core.logging_config import get_logger
from app.schemas import EvaluateRequest
from app.utils.validators import parse_model, require_text_field, validate_max_score

logger = get_logger("routes.evaluation")
bp = Blueprint("evaluation", __name__)


def _evaluation_service():
    return current_app.config["EVALUATION_SERVICE"]


@bp.post("/evaluate")
def evaluate():
    settings = current_app.config["SETTINGS"]
    payload = parse_model(EvaluateRequest)

    question = require_text_field(
        payload.question, "question", settings.max_question_chars, "Question"
    )
    answer = require_text_field(
        payload.answer, "answer", settings.max_answer_chars, "Answer"
    )
    max_score = validate_max_score(payload.max_score, settings)
    rubric = payload.rubric

    result = _evaluation_service().run_text_evaluation(
        question=question,
        answer=answer,
        max_score=max_score,
        rubric=rubric,
        student_id=payload.student_id,
        subject=payload.subject,
    )
    return jsonify({"success": True, "data": result}), 201


@bp.get("/evaluations/<evaluation_id>")
def get_evaluation(evaluation_id: str):
    result = _evaluation_service().get_evaluation(evaluation_id)
    if result is None:
        raise ResourceNotFoundError(f"Evaluation '{evaluation_id}' not found.")
    return jsonify({"success": True, "data": result}), 200