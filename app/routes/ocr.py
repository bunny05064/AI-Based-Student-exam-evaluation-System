"""OCR endpoints: standalone /api/v1/ocr and /api/v1/evaluate/image."""

import json

from flask import Blueprint, current_app, jsonify, request

from app.core.exceptions import ValidationError
from app.core.logging_config import get_logger
from app.utils.validators import require_text_field, validate_max_score

logger = get_logger("routes.ocr")
bp = Blueprint("ocr", __name__)


def _evaluation_service():
    return current_app.config["EVALUATION_SERVICE"]


def _parse_image_form():
    """Extract question/max_score/student/rubric from a multipart form.

    The JSON path validates `student_id` through pydantic (`ge=1`); this path
    has to apply the same rule, otherwise `student_id=0` or `-5` is accepted
    here and rejected by `POST /evaluate` for the identical field.
    """
    settings = current_app.config["SETTINGS"]
    question = require_text_field(
        request.form.get("question"),
        "question",
        settings.max_question_chars,
        "Question",
    )
    max_score = validate_max_score(request.form.get("max_score"), settings)

    student_id = None
    student_id_raw = request.form.get("student_id")
    if student_id_raw:
        try:
            student_id = int(student_id_raw)
        except (TypeError, ValueError):
            raise ValidationError(
                "student_id must be an integer.", details={"student_id": student_id_raw}
            )
        if student_id < 1:
            raise ValidationError(
                "student_id must be a positive integer.",
                details={"student_id": student_id},
            )

    rubric = None
    rubric_raw = request.form.get("rubric")
    if rubric_raw:
        try:
            rubric = json.loads(rubric_raw)
        except json.JSONDecodeError as exc:
            raise ValidationError("rubric must be a valid JSON object.") from exc
        if not isinstance(rubric, dict):
            raise ValidationError("rubric must be a valid JSON object.")

    subject = request.form.get("subject") or None
    if subject is not None:
        subject = subject.strip()[:120] or None

    return question, max_score, student_id, rubric, subject


def _take_file():
    files = request.files
    if not files:
        raise ValidationError("An image/PDF file is required.")
    file = files.get("image") or files.get("file")
    if file is None or not file.filename:
        raise ValidationError("An image/PDF file is required.")
    return file


@bp.post("/ocr")
def ocr():
    """Extract text from an uploaded image/PDF (no evaluation)."""
    file = _take_file()
    data = file.read()
    ocr_service = current_app.config["OCR_SERVICE"]
    result = ocr_service.process(data, file.filename)
    logger.info("OCR endpoint completed | engine=%s", result["engine"])
    return jsonify({"success": True, "data": result}), 200


@bp.post("/evaluate/image")
def evaluate_image():
    question, max_score, student_id, rubric, subject = _parse_image_form()
    file = _take_file()
    data = file.read()

    result = _evaluation_service().evaluate_image(
        question=question,
        image_bytes=data,
        filename=file.filename,
        max_score=max_score,
        rubric=rubric,
        student_id=student_id,
        subject=subject,
    )
    return jsonify({"success": True, "data": result}), 201