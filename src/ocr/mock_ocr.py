"""Mock OCR engine - used ONLY for tests and explicit OCR_ENGINE=mock config.

It does not pretend to be real OCR: the engine name is "mock" and the text is
whatever was configured (default a fixed demo string). Never enabled by
default in the application.
"""

from src.ocr.base_ocr import OCREngine, OCRResult

DEFAULT_MOCK_TEXT = (
    "Newton's second law says that force depends on the mass of an object "
    "and how quickly its velocity changes."
)


class MockOCREngine(OCREngine):
    name = "mock"

    def __init__(self, text: str = DEFAULT_MOCK_TEXT, confidence: float = 0.9) -> None:
        self.text = text
        self.confidence = confidence

    def extract_text(self, data: bytes, is_pdf: bool = False) -> OCRResult:
        return OCRResult(
            text=self.text,
            confidence=self.confidence,
            pages=1,
            engine="mock",
            metadata={"note": "mock OCR - never used in production models"},
        )