"""Custom exception hierarchy for the application.

Every exception has an HTTP status code and a stable machine-readable
``code`` so the global error handlers can produce safe, consistent JSON
without ever exposing internal details to the client.
"""

from typing import Any, Optional


class AppError(Exception):
    """Base exception for all application-level errors."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        details: Optional[Any] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        if code is not None:
            # Lets the error handlers reuse to_dict() for Werkzeug errors,
            # which carry their own codes but not their own class.
            self.code = code

    def to_dict(self) -> dict:
        payload: dict = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


class ValidationError(AppError):
    """Invalid or malformed client input."""

    status_code = 400
    code = "INVALID_INPUT"


class ResourceNotFoundError(AppError):
    """Requested resource does not exist."""

    status_code = 404
    code = "RESOURCE_NOT_FOUND"


class EvaluationError(AppError):
    """The evaluation pipeline could not produce a result."""

    status_code = 500
    code = "EVALUATION_FAILED"


class ModelLoadError(AppError):
    """A model could not be loaded from disk."""

    status_code = 503
    code = "MODEL_LOAD_FAILED"


class ModelInferenceError(AppError):
    """Model inference failed at runtime."""

    status_code = 500
    code = "MODEL_INFERENCE_FAILED"


class InvalidModelStateError(AppError):
    """Model is not in a usable state (e.g. trained model is required)."""

    status_code = 503
    code = "INVALID_MODEL_STATE"


class OCRException(AppError):
    """OCR processing failed for a provided document/image."""

    status_code = 422
    code = "OCR_FAILED"


class FileProcessingError(AppError):
    """Generic file processing failure."""

    status_code = 500
    code = "FILE_PROCESSING_FAILED"


class UnsupportedFileError(FileProcessingError):
    """File type or extension is not supported."""

    status_code = 400
    code = "UNSUPPORTED_FILE"


class FileTooLargeError(FileProcessingError):
    """Uploaded file exceeds the configured size limit."""

    status_code = 413
    code = "FILE_TOO_LARGE"


class DatasetError(AppError):
    """Dataset could not be loaded, validated, or preprocessed."""

    status_code = 500
    code = "DATASET_ERROR"