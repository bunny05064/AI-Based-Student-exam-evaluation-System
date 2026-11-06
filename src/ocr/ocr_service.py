"""OCR service: file validation, engine resolution, PDF text extraction."""

import io
import logging

from app.core.exceptions import OCRException
from app.utils.files import assert_supported_extension, assert_within_size_limit, sniff_image_format
from src.ocr.base_ocr import OCREngine, OCRResult
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.mock_ocr import MockOCREngine
from src.ocr.tesseract_ocr import TesseractEngine

logger = logging.getLogger("src.ocr")

ENGINE_REGISTRY = {
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
    "mock": MockOCREngine,
}


def build_engine(
    engine_name: str, ocr_min_confidence: float = 0.0
) -> OCREngine:
    name = (engine_name or "tesseract").strip().lower()
    if name == "tesseract":
        return TesseractEngine(min_confidence=ocr_min_confidence)
    if name == "easyocr":
        return EasyOCREngine()
    if name == "mock":
        return MockOCREngine()
    raise ValueError(f"Unknown OCR engine: {engine_name}. Known: {', '.join(ENGINE_REGISTRY)}")


class OCRService:
    """Combine upload validation + the configured OCR auto-engine + PDF text path."""

    def __init__(
        self,
        engine: OCREngine,
        max_upload_mb: int,
        min_confidence: float = 0.0,
        max_pdf_pages: int = 20,
    ):
        self.engine = engine
        self.max_upload_mb = max_upload_mb
        self.min_confidence = min_confidence
        self.max_pdf_pages = max_pdf_pages

    def process(self, data: bytes, filename: str) -> dict:
        """Validate and extract text; returns a safe JSON-ready dict."""
        ext = assert_supported_extension(filename)
        assert_within_size_limit(len(data), self.max_upload_mb)

        if ext == "pdf":
            return self._extract_pdf_text(data)._as_dict()

        # Validate that the bytes actually are an image BEFORE running OCR.
        sniff_image_format(data)
        try:
            result = self.engine.extract_text(data, is_pdf=False)
        except OCRException:
            logger.exception("OCR engine failed | engine=%s", getattr(self.engine, "name", "?"))
            raise

        # OCR_MIN_CONFIDENCE is a gate, not advice: grading an image the engine
        # could barely read produces a confidently wrong score.
        if self.min_confidence > 0 and result.confidence < self.min_confidence:
            logger.warning(
                "OCR confidence below threshold | confidence=%s min=%s",
                result.confidence, self.min_confidence,
            )
            raise OCRException(
                "The handwriting could not be read confidently enough "
                f"(confidence {result.confidence:.0%} < {self.min_confidence:.0%}). "
                "Try a clearer, better-lit photo, or type the answer instead.",
                details={
                    "confidence": result.confidence,
                    "min_confidence": self.min_confidence,
                },
            )
        return result._as_dict()

    def _extract_pdf_text(self, data: bytes) -> OCRResult:
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:
            raise OCRException("pypdf is not installed for PDF text extraction.") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = len(reader.pages or [])
            # A 10MB upload can declare thousands of pages; parsing all of them
            # before the answer-length check is an easy way to burn CPU/memory.
            if self.max_pdf_pages > 0 and pages > self.max_pdf_pages:
                raise OCRException(
                    f"The PDF has {pages} pages; at most {self.max_pdf_pages} "
                    "are supported. Upload only the relevant pages.",
                    details={"pages": pages, "max_pdf_pages": self.max_pdf_pages},
                )
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001
                    parts.append("")
            text = " ".join(p.strip() for p in parts).strip()
        except OCRException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRException("Could not parse the uploaded PDF.") from exc

        if not text:
            raise OCRException(
                "No selectable text found in the PDF (scanned/image-only PDFs "
                "are not supported; upload an image instead)."
            )
        confidence = 1.0 if pages else 0.0
        return OCRResult(
            text=text,
            confidence=confidence,
            pages=pages,
            engine="pdf-text-extractor",
            metadata={"hint": "text-layer extraction, not visual OCR"},
        )