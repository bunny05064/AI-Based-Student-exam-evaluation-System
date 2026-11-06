"""App-level OCR service accessor (thin config wiring around src.ocr)."""

import logging

from app.core.exceptions import OCRException
from src.ocr import OCRService, build_engine
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.mock_ocr import MockOCREngine
from src.ocr.tesseract_ocr import TesseractEngine

logger = logging.getLogger("app.services.ocr")

_ENGINE_CLASSES = {
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
    "mock": MockOCREngine,
}


def get_ocr_service(settings) -> OCRService:
    """Build a configured OCRService from application settings.

    Outside production, a requested engine whose dependencies are missing falls
    back to the clearly-labelled `mock` engine with a loud warning - matching
    the "runs out of the box with no Tesseract installed" promise. In
    production it raises instead: silently returning mock text would hand back
    confident scores for text that was never read.
    """
    name = (settings.ocr_engine or "tesseract").strip().lower()
    engine_class = _ENGINE_CLASSES.get(name)
    if engine_class is not None and not engine_class.is_available():
        if settings.flask_env == "production":
            raise OCRException(
                f"OCR_ENGINE={name} was requested but {name} is not installed. "
                "Install requirements-ocr.txt (and the Tesseract system binary) "
                "or set OCR_ENGINE=mock explicitly."
            )
        logger.warning(
            "OCR engine %r is not installed - falling back to the mock engine. "
            "Results are labelled 'mock' and are not real OCR.",
            name,
        )
        engine = build_engine("mock", settings.ocr_min_confidence)
    else:
        engine = build_engine(name, settings.ocr_min_confidence)

    return OCRService(
        engine=engine,
        max_upload_mb=settings.max_upload_mb,
        min_confidence=settings.ocr_min_confidence,
        max_pdf_pages=settings.max_pdf_pages,
    )