"""EasyOCR engine wrapper (bundled models; lazy first-use download)."""

import io

from app.core.exceptions import OCRException
from src.ocr.base_ocr import OCREngine, OCRResult


class EasyOCREngine(OCREngine):
    name = "easyocr"

    @classmethod
    def is_available(cls) -> bool:
        from importlib.util import find_spec

        return find_spec("easyocr") is not None

    def __init__(self, languages: tuple = ("en",)) -> None:
        self.languages = list(languages)
        self._reader = None

    def _load(self):
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # noqa: PLC0415
        except ImportError as exc:
            raise OCRException(
                "EasyOCR is not installed. Install requirements-ocr.txt."
            ) from exc
        self._reader = easyocr.Reader(self.languages)
        return self._reader

    def extract_text(self, data: bytes, is_pdf: bool = False) -> OCRResult:
        if is_pdf:
            raise OCRException(
                "EasyOCR does not natively read PDFs; upload an image instead."
            )
        import numpy as np
        from PIL import Image as PILImage

        try:
            image = PILImage.open(io.BytesIO(data)).convert("RGB")
            array = np.asarray(image)
        except Exception as exc:  # noqa: BLE001
            raise OCRException("Could not read the uploaded image.") from exc

        try:
            reader = self._load()
            results = reader.readtext(array)
        except Exception as exc:  # noqa: BLE001
            raise OCRException("EasyOCR inference failed.") from exc

        if not results:
            raise OCRException("No text could be recognized in the image.")

        text = " ".join(line[1] for line in results).strip()
        confidence = round(
            sum(float(line[2]) for line in results) / len(results), 4
        )
        return OCRResult(
            text=text, confidence=confidence, pages=1, engine="easyocr",
            metadata={"languages": self.languages, "lines": len(results)},
        )