import re

import logfire
from bs4 import BeautifulSoup

from app.ingestion.loaders.document import ParsedDoc

_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
# Block-level tags that start a new paragraph in rendered HTML.
_BLOCK_TAGS = [
    *_HEADING_TAGS, "p", "li", "pre", "blockquote",
    "table", "tr", "dt", "dd", "div", "section", "article", "header", "footer", "title",
]  # fmt: skip
# Temporary marker put at the start of each heading so we can find it after get_text().
_HEADING_MARK_RE = re.compile(r"\x00H([1-6])\x00")


def load_html(file_path: str) -> ParsedDoc:
    """
    Parse HTML with BeautifulSoup into paragraph-separated text.
    Records the page <title> and the heading path (h1 > h2 > ...) of every section.
    """
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            soup = BeautifulSoup(content, "html.parser")

            # 1. Remove Junk (Scripts, Styles, Metadata)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()

            title_tag = soup.find("title")
            title = title_tag.get_text(" ", strip=True) if title_tag else None

            # 2. Mark headings, and block boundaries with blank lines so the chunker can split on paragraphs
            for h in soup.find_all(_HEADING_TAGS):
                h.insert(0, f"\x00H{h.name[1]}\x00")
            for el in soup.find_all(_BLOCK_TAGS):
                el.insert_before("\n\n")
                el.insert_after("\n\n")
            for br in soup.find_all("br"):
                br.replace_with("\n")

            # 3. Extract text (inline tags join naturally, like a browser renders them)
            text = soup.get_text()

            # 4. Clean whitespace and track heading positions
            blocks: list[str] = []
            sections: list[tuple[int, str]] = []
            heading_stack: dict[int, str] = {}
            offset = 0
            for block in re.split(r"\n\s*\n", text):
                lines = [line.strip() for line in block.splitlines()]
                block_clean = "\n".join(line for line in lines if line)
                mark = _HEADING_MARK_RE.match(block_clean)
                block_clean = _HEADING_MARK_RE.sub("", block_clean).strip()
                if not block_clean:
                    continue
                if mark:
                    level = int(mark.group(1))
                    heading_stack = {lvl: t for lvl, t in heading_stack.items() if lvl < level}
                    heading_stack[level] = block_clean
                    sections.append((offset, " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))))
                blocks.append(block_clean)
                offset += len(block_clean) + 2  # + "\n\n"

            if not title and sections:
                title = sections[0][1]

            return ParsedDoc(text="\n\n".join(blocks), title=title or None, sections=sections)
        except Exception as e:
            logfire.error(f"❌ HTML Parse Failed: {e}")
            raise e


def parse_html(file_path: str) -> str:
    """Plain-text view of load_html (kept for callers that only need text)."""
    return load_html(file_path).text
