"""OCR (Optical Character Recognition) with Tesseract.

Tesseract is a system program (brew install tesseract / apt-get install tesseract-ocr);
pytesseract only calls it. If it isn't installed, OCR is skipped with a warning.
"""

import os
import shutil
from functools import lru_cache

import logfire

from app.config import settings

# Common install locations, used when `tesseract` isn't on PATH (e.g. Homebrew not in PATH).
_TESSERACT_CANDIDATES = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"]


@lru_cache(maxsize=1)
def _tesseract_cmd() -> str | None:
    if settings.TESSERACT_CMD:
        return settings.TESSERACT_CMD
    found = shutil.which("tesseract")
    if found:
        return found
    return next((p for p in _TESSERACT_CANDIDATES if os.path.exists(p)), None)


def ocr_available() -> bool:
    """True if OCR is enabled and the Tesseract program can be found."""
    if not settings.OCR_ENABLED:
        return False
    if _tesseract_cmd() is None:
        logfire.warning("OCR enabled but Tesseract not found (brew install tesseract); skipping OCR.")
        return False
    return True


def ocr_image(image) -> str:
    """OCR a PIL image. Converted to grayscale, which helps Tesseract on colour scans."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd()
    if image.mode not in ("L", "1"):
        image = image.convert("L")
    return pytesseract.image_to_string(image, lang=settings.OCR_LANG)


def ocr_pdf_pages(file_path: str, page_numbers: list[int]) -> dict[int, str]:
    """Render the given 1-based PDF pages at OCR_DPI and OCR them. Returns {page: text}."""
    import pymupdf
    from PIL import Image

    results: dict[int, str] = {}
    with logfire.span("OCR PDF pages", filename=file_path, pages=len(page_numbers), dpi=settings.OCR_DPI):
        with pymupdf.open(file_path) as doc:
            for page_num in page_numbers:
                pix = doc[page_num - 1].get_pixmap(dpi=settings.OCR_DPI, colorspace=pymupdf.csGRAY)
                image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                text = ocr_image(image)
                if text.strip():
                    results[page_num] = text
        logfire.info(f"OCR recovered text on {len(results)}/{len(page_numbers)} pages.")
    return results
