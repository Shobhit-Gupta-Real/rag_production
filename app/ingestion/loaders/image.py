import logfire
from PIL import Image, ImageSequence

from app.ingestion.loaders.document import ParsedDoc
from app.ingestion.loaders.ocr import ocr_available, ocr_image

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"}


def load_image(file_path: str) -> ParsedDoc:
    """OCR an image file. Multi-frame images (e.g. multi-page TIFF) become one page per frame."""
    with logfire.span("🖼️ Image OCR", filename=file_path):
        if not ocr_available():
            return ParsedDoc(text="", extraction="ocr")

        parts: list[str] = []
        pages: list[tuple[int, int]] = []
        offset = 0
        with Image.open(file_path) as img:
            frames = list(ImageSequence.Iterator(img))
            for i, frame in enumerate(frames, start=1):
                text = ocr_image(frame.copy()).strip()
                if not text:
                    continue
                if len(frames) > 1:
                    pages.append((offset, i))
                parts.append(text)
                offset += len(text) + 2  # + "\n\n"

        full_text = "\n\n".join(parts)
        logfire.info(f"OCR extracted {len(full_text)} characters from {len(frames)} frame(s).")
        return ParsedDoc(text=full_text, pages=pages, extraction="ocr")
