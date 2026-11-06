"""Global error handlers that convert exceptions into safe JSON responses.

Rules:
- Application exceptions (AppError) keep their stable code/status.
- Werkzeug HTTP exceptions are wrapped in the same JSON shape.
- Unknown exceptions become a generic 500 response; internals are never leaked.
"""

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from app.core.exceptions import AppError
from app.core.logging_config import get_logger

logger = get_logger("errors")


def _json_response(status: int, code: str, message: str, details=None) -> tuple:
    # Routed through AppError.to_dict so the wire shape has exactly one
    # definition; the HTTP status is passed separately to jsonify.
    error = AppError(message, details, code=code)
    return jsonify({"success": False, "error": error.to_dict()}), status


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(AppError)
    def handle_app_error(exc: AppError):
        logger.error(
            "App error | code=%s status=%s message='%s'",
            exc.code,
            exc.status_code,
            exc.message,
        )
        return _json_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        code = str(exc.name or "HTTP_ERROR").upper().replace(" ", "_")
        message = (getattr(exc, "description", None) or exc.name or "HTTP error")
        return _json_response(exc.code or 500, code, message)

    @app.errorhandler(Exception)
    def handle_unexpected(_exc: Exception):
        # Log the full traceback server-side only.
        logger.exception("Unhandled exception occurred")
        return _json_response(
            500, "INTERNAL_ERROR", "An unexpected server error occurred."
        )

    # Ensure validation errors from Werkzeug (e.g. payload too large) are JSON.
    @app.errorhandler(413)
    def handle_too_large(_exc):
        return _json_response(413, "FILE_TOO_LARGE", "Uploaded file is too large.")

    @app.errorhandler(400)
    def handle_bad_request(_exc):
        return _json_response(400, "BAD_REQUEST", "Malformed request.")