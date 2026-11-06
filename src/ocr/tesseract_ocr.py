"""Tesseract OCR engine (pytesseract wrapper).

Requires the Tesseract system binary. If it is missing, extract_text raises
OCRException with an actionable message instead of pretending to work.
"""

import io

from app.core.exceptions import OCRException
from src.ocr.base_ocr import OCREngine, OCRResult


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, min_confidence: float = 0.0) -> None:
        self.min_confidence = min_confidence

    @classmethod
    def is_available(cls) -> bool:
        """Both the pytesseract binding and the system binary must be present."""
        try:
            import pytesseract  # noqa: PLC0415
        except ImportError:
            return False
        try:
            return bool(pytesseract.get_tesseract_version())
        except Exception:  # noqa: BLE001
            return False

    def extract_text(self, data: bytes, is_pdf: bool = False) -> OCRResult:
        if is_pdf:
            raise OCRException(
                "Tesseract PDF conversion needs poppler/pdf2image; use an "
                "image upload or the built-in PDF text extraction."
            )
        try:
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise OCRException(
                "Tesseract OCR is not installed. Install requirements-ocr.txt, "
                "or set OCR_ENGINE=mock for development/testing."
            ) from exc

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise OCRException("Could not read the uploaded image.") from exc

        try:
            details = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as exc:  # noqa: BLE001
            raise OCRException(
                "Tesseract failed to process the image. Is the Tesseract "
                "system binary installed and on PATH?"
            ) from exc

        # Pair each word with ITS OWN confidence before filtering: zipping the
        # unfiltered `conf` list against the filtered `words` list shifted every
        # confidence onto the wrong word as soon as one blank token appeared.
        words: list[str] = []
        confidences: list[float] = []
        raw_conf = details.get("conf", []) or []
        for index, word in enumerate(details.get("text", []) or []):
            if not word or not word.strip():
                continue
            words.append(word.strip())
            if index >= len(raw_conf):
                continue
            raw = raw_conf[index]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            # -1 is Tesseract's "not a real word" sentinel (whitespace/blocks).
            if value < 0:
                continue
            confidences.append(min(value / 100.0, 1.0))

        text = " ".join(words).strip()
        confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        if not text:
            raise OCRException(
                "No text could be recognized in the image.", details={"confidence": confidence}
            )
        return OCRResult(
            text=text, confidence=confidence, pages=1, engine="tesseract",
            metadata={"min_confidence": self.min_confidence},
        )

    def __bool__(self) -> bool:
        return True