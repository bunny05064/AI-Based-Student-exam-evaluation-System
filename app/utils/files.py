"""Secure upload handling: extension allow-list, size guard, content sniffing."""

import re
from pathlib import Path

from app.core.exceptions import FileTooLargeError, UnsupportedFileError

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"}
ALLOWED_DOCUMENT_EXTENSIONS = {"pdf"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS

_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def secure_filename(filename: str) -> str:
    """Return a safe basename derived from a user-provided filename."""
    name = Path(filename or "file").name
    name = _UNSAFE_FILENAME_RE.sub("_", name)
    return name[:120] or "file"


def extension_of(filename: str) -> str:
    return Path(filename or "").suffix.lstrip(".").lower()


def assert_supported_extension(filename: str) -> str:
    ext = extension_of(filename)
    if ext not in ALLOWED_EXTENSIONS or len(ext) == 0:
        raise UnsupportedFileError(
            f"Unsupported file type '.{ext}'. Allowed: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
            + "."
        )
    return ext


def assert_within_size_limit(size_bytes: int, max_upload_mb: int) -> None:
    limit = max_upload_mb * 1024 * 1024
    if size_bytes <= 0:
        raise UnsupportedFileError("Uploaded file is empty.")
    if size_bytes > limit:
        raise FileTooLargeError(
            f"File exceeds the maximum upload size of {max_upload_mb} MB.",
            details={"max_upload_mb": max_upload_mb},
        )


def sniff_image_format(data: bytes) -> str:
    """Validate image bytes and return normalized extension. Raises on bad data."""
    from PIL import Image
    import io

    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").lower()
    except Exception:
        raise UnsupportedFileError("Uploaded file is not a valid image.")
    if fmt in {"jpeg", "jpg"}:
        return "jpeg"
    return fmt if fmt in ALLOWED_IMAGE_EXTENSIONS else "jpg"