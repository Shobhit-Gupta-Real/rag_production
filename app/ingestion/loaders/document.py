from dataclasses import dataclass, field


@dataclass
class ParsedDoc:
    """Text extracted from one file, plus position markers used to enrich chunk metadata.

    Markers are (char offset into `text`, value) pairs in ascending offset order.
    A chunk inherits the last section / page marker at or before its start offset.
    """

    text: str
    title: str | None = None
    sections: list[tuple[int, str]] = field(default_factory=list)  # heading path, e.g. "HPA > How Does HPA Work?"
    pages: list[tuple[int, int]] = field(default_factory=list)  # page (PDF) or slide (PPTX) number
    extraction: str = "text"  # "text" (digital text layer) or "ocr" (whole document recognised from images)
    ocr_pages: set[int] = field(default_factory=set)  # pages recovered by OCR in an otherwise digital PDF

    def extraction_for_page(self, page: int | None) -> str:
        """How a chunk starting on this page was extracted: "ocr" or "text"."""
        if self.extraction == "ocr" or (page is not None and page in self.ocr_pages):
            return "ocr"
        return "text"
