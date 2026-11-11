import re

import logfire

from app.ingestion.loaders.document import ParsedDoc

# Markdown "# Heading" lines.
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
# A line made only of = or - (3+), used as an underline under a heading.
_UNDERLINE_RE = re.compile(r"^(=|-){3,}$")


def load_text(file_path: str) -> ParsedDoc:
    """
    Parse plain text files. Detects Markdown (# Heading) and underlined
    (Heading / ======) headings as sections.
    """
    with logfire.span("📄 Text Parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            logfire.error(f"❌ Text Parse Failed: {e}")
            raise e

    lines = text.splitlines(keepends=True)
    sections: list[tuple[int, str]] = []
    offset = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        heading = None
        # Must start at column 0: indented "# ..." is a code comment, not a heading.
        md = _MD_HEADING_RE.match(line.rstrip())
        if md:
            heading = md.group(1)
        elif stripped and not _UNDERLINE_RE.match(stripped) and i + 1 < len(lines):
            # Underlined heading, only if it starts a block (file start, blank line or rule above it),
            # so a code line sitting above a "-----" divider isn't mistaken for a heading.
            prev = lines[i - 1].strip() if i > 0 else ""
            if _UNDERLINE_RE.match(lines[i + 1].strip()) and (not prev or _UNDERLINE_RE.match(prev)):
                heading = stripped
        if heading:
            sections.append((offset, heading))
        offset += len(line)

    title = sections[0][1] if sections else None
    return ParsedDoc(text=text, title=title, sections=sections)


def parse_text(file_path: str) -> str:
    """Plain-text view of load_text (kept for callers that only need text)."""
    return load_text(file_path).text
