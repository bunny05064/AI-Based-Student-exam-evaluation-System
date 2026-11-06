"""OCR package: modular, swappable OCR engines behind one service."""

from src.ocr.base_ocr import OCREngine, OCRResult
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.mock_ocr import MockOCREngine
from src.ocr.ocr_service import ENGINE_REGISTRY, OCRService, build_engine
from src.ocr.tesseract_ocr import TesseractEngine

__all__ = [
    "OCREngine",
    "OCRResult",
    "EasyOCREngine",
    "MockOCREngine",
    "TesseractEngine",
    "ENGINE_REGISTRY",
    "OCRService",
    "build_engine",
]