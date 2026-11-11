import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.sparse import sparse_embed_query

# Must match the named vectors created at ingestion (app/ingestion/processor.py).
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
# Candidates each search side contributes before RRF fusion.
_PREFETCH_MIN = 30
# RRF constant: score = Σ 1 / (k + rank). Qdrant defaults to k=2, which lets a single #1 rank
# dominate; 60 is the standard value and rewards chunks ranked well by both searches.
_RRF_K = 60

# Initialize Qdrant Client
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _search_enterprise_knowledge(query: str, limit: int = 8):
    """Hybrid search (dense + BM25, fused with RRF) with retry logic."""
    dense_q = embed_query(query)
    try:
        sparse_q = sparse_embed_query(query)
    except Exception as e:
        logfire.warning(f"BM25 query encoding failed ({e}); using dense-only search.")
        sparse_q = None

    if sparse_q is None or not sparse_q.indices:
        # No usable keywords (e.g. query is all stopwords) → dense only.
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=dense_q,
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
        )
    else:
        # Each side returns its own top-N; RRF merges them by rank: score = Σ 1 / (k + rank).
        prefetch_limit = max(_PREFETCH_MIN, limit * 2)
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_q, using=DENSE_VECTOR, limit=prefetch_limit),
                models.Prefetch(query=sparse_q, using=SPARSE_VECTOR, limit=prefetch_limit),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=_RRF_K)),
            limit=limit,
            with_payload=True,
        )

    results = []
    for res in response.points:
        results.append(
            {"content": res.payload.get("text", ""), "source": res.payload.get("source", "Unknown"), "score": res.score}
        )

    return results


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs hybrid (dense + BM25 keyword) search in the enterprise knowledge base.
    Results are fused with Reciprocal Rank Fusion inside Qdrant. Retries transient failures
    and gracefully degrades to an empty result set on persistent failure.
    """
    try:
        return _search_enterprise_knowledge(query, limit=limit)
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed after retries: {e}")
        return []
