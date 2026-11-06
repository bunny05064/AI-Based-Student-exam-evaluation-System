"""Student endpoints: creation, listing and progress dashboard data."""

from flask import Blueprint, current_app, jsonify

from app.core.logging_config import get_logger
from app.schemas import StudentCreateRequest
from app.utils.validators import parse_model

logger = get_logger("routes.student")
bp = Blueprint("student", __name__)


def _evaluation_service():
    return current_app.config["EVALUATION_SERVICE"]


@bp.post("/student")
def create_student():
    payload = parse_model(StudentCreateRequest)

    student, created = _evaluation_service().get_or_create_student(payload.name)
    logger.info("Student ensured | id=%s name=%s created=%s", student.id, student.name, created)
    # 201 on create, 200 when an existing student was matched by name.
    return jsonify({"success": True, "data": student.to_dict()}), (201 if created else 200)


@bp.get("/students")
def list_students():
    """Recent students, so the UI never requires guessing a numeric id."""
    students = _evaluation_service().list_students(limit=50)
    return jsonify({"success": True, "data": students}), 200


@bp.get("/student/<int:student_id>/progress")
def student_progress(student_id: int):
    progress = _evaluation_service().student_progress(student_id)
    return jsonify({"success": True, "data": progress}), 200