from app.ingestion.loaders.document import ParsedDoc

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "gif"}
SUPPORTED_EXTENSIONS = {"pdf", "html", "htm", "txt", "docx", "pptx"} | _IMAGE_EXTENSIONS


def load_document(file_path: str) -> ParsedDoc | None:
    """Parse a file into a ParsedDoc by extension. Returns None for unsupported types."""
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from app.ingestion.loaders.pdf import load_pdf

        return load_pdf(file_path)
    if ext in ("html", "htm"):
        from app.ingestion.loaders.html import load_html

        return load_html(file_path)
    if ext == "txt":
        from app.ingestion.loaders.text import load_text

        return load_text(file_path)
    if ext in ("docx", "pptx"):
        # Imported lazily: unstructured is heavy.
        from app.ingestion.loaders.office import load_office

        return load_office(file_path)
    if ext in _IMAGE_EXTENSIONS:
        from app.ingestion.loaders.image import load_image

        return load_image(file_path)
    return None
