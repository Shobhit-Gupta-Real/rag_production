import logfire
from unstructured.partition.auto import partition

from app.ingestion.loaders.document import ParsedDoc


def load_office(file_path: str) -> ParsedDoc:
    """
    Parse Office documents (.docx, .pptx) using the Unstructured library.
    Records Title elements as sections and slide/page numbers when available.
    """
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            # Unstructured automatically detects if it's docx or pptx
            elements = partition(filename=file_path)

            # Blank line between elements (titles, paragraphs, list items) so the chunker sees paragraphs
            parts: list[str] = []
            sections: list[tuple[int, str]] = []
            pages: list[tuple[int, int]] = []
            offset = 0
            for el in elements:
                text = str(el).strip()
                if not text:
                    continue
                if getattr(el, "category", None) == "Title":
                    sections.append((offset, text))
                page = getattr(el.metadata, "page_number", None)
                if page is not None and (not pages or pages[-1][1] != page):
                    pages.append((offset, page))
                parts.append(text)
                offset += len(text) + 2  # + "\n\n"
            full_text = "\n\n".join(parts)

            if not full_text.strip():
                logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

            title = sections[0][1] if sections else None
            return ParsedDoc(text=full_text, title=title, sections=sections, pages=pages)
        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e


def parse_office(file_path: str) -> str:
    """Plain-text view of load_office (kept for callers that only need text)."""
    return load_office(file_path).text
