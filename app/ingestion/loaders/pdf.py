import logfire
from pypdf import PdfReader

from app.config import settings
from app.ingestion.loaders.document import ParsedDoc
from app.ingestion.loaders.ocr import ocr_available, ocr_pdf_pages


def load_pdf(file_path: str) -> ParsedDoc:
    """
    Extract text from a PDF locally using pypdf, keeping page numbers.
    Pages with (almost) no text fall back to pdfplumber, then to OCR (scanned / image-only pages).
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")

            page_texts: dict[int, str] = {}  # 1-based page number -> text
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if len(text.strip()) >= settings.OCR_MIN_CHARS:
                    page_texts[i + 1] = text
                else:
                    blank_pages.append(i + 1)

            # Fallback: use pdfplumber for any pages pypdf returned blank
            if blank_pages:
                logfire.info(f"pypdf returned blank on pages {blank_pages} — retrying with pdfplumber.")
                try:
                    import pdfplumber

                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""
                            if len(fallback_text.strip()) >= settings.OCR_MIN_CHARS:
                                page_texts[page_num] = fallback_text
                except Exception as plumber_err:
                    logfire.warning(f"pdfplumber fallback failed: {plumber_err}")

            # OCR: pages that are still (almost) empty are likely scanned images
            ocr_pages: set[int] = set()
            still_blank = [p for p in blank_pages if p not in page_texts]
            if still_blank and ocr_available():
                logfire.info(f"Running OCR on {len(still_blank)} page(s) with no text layer.")
                try:
                    for page_num, ocr_text in ocr_pdf_pages(file_path, still_blank).items():
                        page_texts[page_num] = ocr_text
                        ocr_pages.add(page_num)
                except Exception as ocr_err:
                    logfire.warning(f"OCR failed for {file_path}: {ocr_err}")

            # Blank line between pages so page boundaries are preferred split points; keep page order
            parts: list[str] = []
            pages: list[tuple[int, int]] = []
            offset = 0
            for page_num in sorted(page_texts):
                pages.append((offset, page_num))
                parts.append(page_texts[page_num])
                offset += len(page_texts[page_num]) + 2  # + "\n\n"
            full_text = "\n\n".join(parts)

            if not full_text.strip():
                logfire.warning(f"No text extracted from {file_path}. File may be fully image-based.")
            else:
                logfire.info(f"Extracted {len(full_text)} characters from {file_path}.")

            title = None
            try:
                title = (reader.metadata.title or "").strip() if reader.metadata else None
            except Exception:
                pass

            extraction = "ocr" if ocr_pages and len(ocr_pages) == len(page_texts) else "text"
            return ParsedDoc(
                text=full_text, title=title or None, pages=pages, extraction=extraction, ocr_pages=ocr_pages
            )

        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise


def parse_pdf(file_path: str) -> str:
    """Plain-text view of load_pdf (kept for callers that only need text)."""
    return load_pdf(file_path).text
