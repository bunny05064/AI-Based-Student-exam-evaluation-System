"""Shared request validators used by every API route."""

from typing import Any, Mapping, Optional

from flask import request
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import ValidationError
from app.utils.text import clean_text, is_empty


def json_object_body() -> Mapping[str, Any]:
    """Return the request body as a JSON object.

    `request.get_json(silent=True)` returns `None` for malformed JSON but still
    hands back a *list* or *scalar* for well-formed non-objects. Unpacking
    those with `**` raises `TypeError` -> a 500 for what is plainly a client
    mistake, so the shape is checked here and reported as a 400.
    """
    body = request.get_json(silent=True)
    if body is None:
        raise ValidationError("Request body must be a JSON object.")
    if not isinstance(body, dict):
        raise ValidationError(
            "Request body must be a JSON object.",
            details={"received_type": type(body).__name__},
        )
    return body


def parse_model(schema: type[BaseModel]):
    """Validate the request body against a pydantic schema."""
    body = json_object_body()
    try:
        return schema(**body)
    except PydanticValidationError as exc:
        raise ValidationError(
            "Invalid request body.", details={"errors": exc.errors(include_url=False)}
        ) from exc


def require_text_field(
    value: Optional[str], field_name: str, max_chars: int, label: str
) -> str:
    if value is None:
        raise ValidationError(f"{label} is required.")
    cleaned = clean_text(value)
    if is_empty(cleaned):
        raise ValidationError(f"{label} cannot be empty.")
    if len(cleaned) > max_chars:
        raise ValidationError(
            f"{label} exceeds the maximum length of {max_chars} characters.",
            details={"max_chars": max_chars},
        )
    return cleaned


def validate_max_score(value, settings, required: bool = False) -> int:
    """Validate max_score is a positive integer within allowed bounds."""
    if value is None:
        if required:
            raise ValidationError("max_score is required.")
        return settings.max_score_default
    try:
        score = int(value)
    except (TypeError, ValueError):
        raise ValidationError("max_score must be an integer.", details={"max_score": value})
    if score <= 0:
        raise ValidationError("max_score must be a positive integer.")
    if score > settings.max_score_allowed:
        raise ValidationError(
            f"max_score cannot exceed {settings.max_score_allowed}.",
            details={"max_allowed": settings.max_score_allowed},
        )
    return score