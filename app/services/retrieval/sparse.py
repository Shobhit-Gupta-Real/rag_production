"""BM25 sparse vectors for hybrid search (keyword side).

Our code does tokenize → stopwords → stem → BM25 TF weight → hash (fastembed, local, no API).
Qdrant stores the vectors in an inverted index and applies IDF at query time (Modifier.IDF).
Ingestion and queries must use the same model + settings; changing them requires a re-ingest.
"""

import os

import logfire
from qdrant_client.http import models

BM25_MODEL = "Qdrant/bm25"
# Average chunk length in BM25 tokens (after stopwords/stemming), measured on our
# 450-token chunks (~160). fastembed's default of 256 would under-penalise long chunks.
BM25_AVG_LEN = 160.0
# Persistent cache (fastembed defaults to a temp dir that the OS may clear).
_CACHE_DIR = os.path.expanduser("~/.cache/fastembed")

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding

        logfire.info(f"Loading BM25 sparse model ({BM25_MODEL}, avg_len={BM25_AVG_LEN}).")
        _model = SparseTextEmbedding(BM25_MODEL, cache_dir=_CACHE_DIR, avg_len=BM25_AVG_LEN)
    return _model


def _to_qdrant(emb) -> models.SparseVector:
    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def sparse_embed_texts(texts: list[str]) -> list[models.SparseVector]:
    """BM25 document vectors (TF-weighted; IDF is applied by Qdrant)."""
    with logfire.span("BM25 sparse embedding", size=len(texts)):
        return [_to_qdrant(e) for e in _get_model().embed(texts)]


def sparse_embed_query(query: str) -> models.SparseVector:
    """BM25 query vector (each unique query term has weight 1)."""
    return _to_qdrant(next(iter(_get_model().query_embed(query))))
