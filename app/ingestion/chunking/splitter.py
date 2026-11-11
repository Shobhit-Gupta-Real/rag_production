from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import List, TypeVar

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from app.ingestion.loaders.document import ParsedDoc

# Sizes are in Jina v3 tokens (the embedding model's own tokenizer).
CHUNK_SIZE_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 70  # ~15% overlap so facts on a boundary survive whole in one chunk

# Try paragraph → line → sentence → word → character, until a piece fits.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
# Used only if the tokenizer can't be loaded (rough English average).
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _get_splitter() -> TextSplitter:
    """Build the recursive splitter once, measuring length in Jina v3 tokens."""
    common = {"separators": _SEPARATORS, "keep_separator": "end"}
    try:
        from transformers import AutoTokenizer

        from app.services.retrieval.embedding import JINA_HF_MODEL, JINA_HF_REVISION

        tokenizer = AutoTokenizer.from_pretrained(JINA_HF_MODEL, revision=JINA_HF_REVISION)
        # We only count tokens here; silence the ">8192 tokens" warning on long documents.
        tokenizer.model_max_length = 10**9
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            **common,
        )
    except Exception as e:
        logfire.warning(f"Jina tokenizer unavailable ({e}); falling back to character-based chunk sizes.")
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS * _CHARS_PER_TOKEN,
            chunk_overlap=CHUNK_OVERLAP_TOKENS * _CHARS_PER_TOKEN,
            **common,
        )


def chunk_text(text: str) -> List[str]:
    """
    Split text into ~450-token chunks with ~70-token overlap.
    Recursive splitting enforces a hard cap even when the text has no paragraph breaks.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        chunks = [c for c in _get_splitter().split_text(text) if c.strip()]
        logfire.info(f"✅ Generated {len(chunks)} chunks")
        return chunks


@dataclass
class Chunk:
    text: str
    start_index: int  # char offset of the chunk in the document text (-1 if unknown)
    section: str | None = None
    page: int | None = None


_T = TypeVar("_T")


def _marker_at(markers: list[tuple[int, _T]], pos: int) -> _T | None:
    """Value of the last marker at or before pos."""
    if pos < 0 or not markers:
        return None
    i = bisect_right([offset for offset, _ in markers], pos)
    return markers[i - 1][1] if i else None


def chunk_document(doc: ParsedDoc) -> List[Chunk]:
    """Chunk a parsed document and attach the section heading and page each chunk starts in."""
    with logfire.span("✂️ Document Chunking", text_length=len(doc.text)):
        if not doc.text.strip():
            return []

        # Locate each chunk ourselves: LangChain's add_start_index subtracts the overlap (in tokens)
        # from a char offset, so with a token length function its search starts too late and misses.
        chunks = []
        search_from = 0
        for text in _get_splitter().split_text(doc.text):
            if not text.strip():
                continue
            start = doc.text.find(text, search_from)
            if start == -1:
                start = doc.text.find(text)
            if start != -1:
                search_from = start + 1
            chunks.append(
                Chunk(
                    text=text,
                    start_index=start,
                    section=_marker_at(doc.sections, start),
                    page=_marker_at(doc.pages, start),
                )
            )
        logfire.info(f"✅ Generated {len(chunks)} chunks")
        return chunks
