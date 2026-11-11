import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# logfire must be configured before app module imports so spans from
# chunking/loaders/embedding are captured from the start.
import logfire
from app.config import settings

_logfire_base_url = settings.LOGFIRE_BASE_URL
if not _logfire_base_url and settings.LOGFIRE_TOKEN:
    if settings.LOGFIRE_TOKEN.startswith("pylf_v2_eu_"):
        _logfire_base_url = "https://logfire-eu.pydantic.dev"

if settings.LOGFIRE_TOKEN:
    logfire.configure(
        token=settings.LOGFIRE_TOKEN,
        service_name="enterprise-ingestion-service",
        advanced=logfire.AdvancedOptions(base_url=_logfire_base_url) if _logfire_base_url else None,
    )

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingestion.chunking.splitter import chunk_document
from app.ingestion.loaders import load_document
from app.services.retrieval.embedding import embed_texts_with_provider, get_embedding_dim
from app.services.retrieval.sparse import sparse_embed_texts

# Named vectors on each point: dense (meaning) + sparse BM25 (keywords) for hybrid search.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Fixed namespace so the same (source_type, file, chunk_index) always maps to the same point ID.
# Re-ingesting overwrites points in place instead of creating duplicates. Never change this value.
_POINT_ID_NAMESPACE = uuid.UUID("6f1c2a4e-9b3d-4c8a-a1e7-2d5f8b0c9e31")

# Payload fields used in filters (ingestion dedup/prune, and retrieval filters).
_PAYLOAD_INDEXES = {
    "source": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "file_hash": models.PayloadSchemaType.KEYWORD,
}

_UPSERT_BATCH_SIZE = 256

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _point_id(source_type: str, filename: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{source_type}/{filename}#{chunk_index}"))


def _file_hash(file_path: str) -> str:
    """SHA-256 of the file's bytes — changes only when the content changes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source_filter(source_type: str, filename: str) -> list[models.FieldCondition]:
    return [
        models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
        models.FieldCondition(key="source", match=models.MatchValue(value=filename)),
    ]


def _is_up_to_date(source_type: str, filename: str, file_hash: str) -> bool:
    """True if Qdrant already holds a complete set of points for this exact file content."""
    same_hash = models.Filter(
        must=[
            *_source_filter(source_type, filename),
            models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash)),
        ]
    )
    points, _ = qdrant_client.scroll(
        settings.QDRANT_COLLECTION, scroll_filter=same_hash, limit=1, with_payload=["chunk_count"]
    )
    if not points:
        return False
    expected = points[0].payload.get("chunk_count")
    # Guard against a previous run that crashed mid-upsert: only skip if every chunk is present.
    found = qdrant_client.count(settings.QDRANT_COLLECTION, count_filter=same_hash, exact=True).count
    return expected is not None and found == expected


def _delete_points(flt: models.Filter) -> None:
    qdrant_client.delete(settings.QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=flt), wait=True)


def _processed_path(source_type: str, filename: str) -> str:
    return os.path.join(PROCESSED_DATA_DIR, source_type, f"{filename}.json")


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    dest = _processed_path(source_type, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def remove_file_from_index(source_type: str, filename: str) -> None:
    """Delete a file's points from Qdrant and its local processed JSON."""
    _delete_points(models.Filter(must=_source_filter(source_type, filename)))
    try:
        os.remove(_processed_path(source_type, filename))
    except FileNotFoundError:
        pass


def _ensure_collection() -> None:
    """Create the hybrid collection if missing (dense dim probed at runtime) and its payload indexes."""
    if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
        dim = get_embedding_dim()
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            # IDF modifier: Qdrant computes word rarity across the collection, completing BM25.
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        logfire.info(f"Created collection '{settings.QDRANT_COLLECTION}' ({dim}-dim dense + BM25 sparse).")
    else:
        params = qdrant_client.get_collection(settings.QDRANT_COLLECTION).config.params
        has_dense = isinstance(params.vectors, dict) and DENSE_VECTOR in params.vectors
        has_sparse = bool(params.sparse_vectors) and SPARSE_VECTOR in params.sparse_vectors
        if not (has_dense and has_sparse):
            raise RuntimeError(
                f"Collection '{settings.QDRANT_COLLECTION}' uses the old single-vector schema. "
                "Re-run ingestion with --wipe to recreate it for hybrid search."
            )

    # Idempotent: creating an index that already exists is a no-op.
    for field, schema in _PAYLOAD_INDEXES.items():
        qdrant_client.create_payload_index(settings.QDRANT_COLLECTION, field_name=field, field_schema=schema, wait=True)


# ── Ingestion ──────────────────────────────────────────────────────────────────


def process_file(file_path: str, filename: str, source_type: str, force: bool = False) -> str:
    """
    Parse → chunk → save locally → embed → index in Qdrant, idempotently.

    Skips files whose content hash is already fully indexed. Changed files get their
    points overwritten (deterministic IDs) and leftover chunks from the old version deleted.
    Returns a status: "indexed", "unchanged", "empty", "unsupported" or "failed".
    """
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            file_hash = _file_hash(file_path)
            if not force and _is_up_to_date(source_type, filename, file_hash):
                logfire.info(f"Unchanged, skipping: {filename}")
                return "unchanged"

            # 1. Extract text + structure based on file extension
            doc = load_document(file_path)
            if doc is None:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return "unsupported"

            # 2. Chunk text (each chunk knows its section and page)
            chunks = chunk_document(doc) if doc.text.strip() else []
            if not chunks:
                logfire.warning(f"No text extracted from {filename} — removing any old points.")
                remove_file_from_index(source_type, filename)
                return "empty"

            title = doc.title or Path(filename).stem
            file_modified_at = datetime.fromtimestamp(os.path.getmtime(file_path), timezone.utc).isoformat()
            ingested_at = datetime.now(timezone.utc).isoformat()

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "title": title,
                "file_hash": file_hash,
                "chunks": [
                    {
                        "chunk_index": i,
                        "section": c.section,
                        "page": c.page,
                        "extraction": doc.extraction_for_page(c.page),
                        "text": c.text,
                    }
                    for i, c in enumerate(chunks)
                ],
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                texts = [c.text for c in chunks]
                use_local = source_type in settings.embed_local_source_types
                embeddings, provider = embed_texts_with_provider(texts, local=use_local)
                sparse_vectors = sparse_embed_texts(texts)
                points = [
                    models.PointStruct(
                        id=_point_id(source_type, filename, i),
                        vector={DENSE_VECTOR: vector, SPARSE_VECTOR: sparse},
                        payload={
                            "text": chunk.text,
                            "source": filename,
                            "source_type": source_type,
                            "title": title,
                            "section": chunk.section,
                            "page": chunk.page,
                            "extraction": doc.extraction_for_page(chunk.page),
                            "chunk_index": i,
                            "chunk_count": len(chunks),
                            "file_hash": file_hash,
                            "embedding_provider": provider,
                            "file_modified_at": file_modified_at,
                            "ingested_at": ingested_at,
                        },
                    )
                    for i, (chunk, vector, sparse) in enumerate(zip(chunks, embeddings, sparse_vectors))
                ]

                for i in range(0, len(points), _UPSERT_BATCH_SIZE):
                    qdrant_client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=points[i : i + _UPSERT_BATCH_SIZE],
                        wait=True,
                    )

                # Remove points from an older version of this file (e.g. it used to have more chunks).
                _delete_points(
                    models.Filter(
                        must=_source_filter(source_type, filename),
                        must_not=[models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash))],
                    )
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename} (embedded via {provider}).")
                return "indexed"

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return "failed"


def prune_removed_files(source_type: str, present_files: set[str]) -> list[str]:
    """Delete points (and processed JSON) for files of this source_type no longer on disk."""
    indexed: set[str] = set()
    offset = None
    flt = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type))])
    while True:
        points, offset = qdrant_client.scroll(
            settings.QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=1000,
            offset=offset,
            with_payload=["source"],
        )
        indexed.update(p.payload.get("source") for p in points)
        if offset is None:
            break

    removed = sorted(s for s in indexed - present_files if s)
    for filename in removed:
        remove_file_from_index(source_type, filename)
        logfire.info(f"Pruned removed file: {source_type}/{filename}")
    return removed


def process_directory(dir_path: str, source_type: str, force: bool = False, prune: bool = True) -> dict:
    """Process every file in a directory, then prune points for files that were deleted."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = sorted(f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
        logfire.info(f"Found {len(files)} files in {dir_path}.")

        stats: dict[str, int] = {}
        for filename in files:
            status = process_file(os.path.join(dir_path, filename), filename, source_type, force=force)
            stats[status] = stats.get(status, 0) + 1

        if prune:
            stats["pruned"] = len(prune_removed_files(source_type, set(files)))

        logfire.info(f"Directory '{dir_path}' ({source_type}) done: {stats}")
        return stats


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str = None,
    wipe: bool = False,
    force: bool = False,
    prune: bool = True,
):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    --wipe   drop and recreate the Qdrant collection before ingestion.
    --force  re-embed every file even if unchanged.
    --no-prune  keep points for files that are no longer on disk.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        _ensure_collection()

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = "true" if "true" in base_name else "noisy" if "noisy" in base_name else "general"
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            return {source_type: process_directory(base_dir, source_type, force=force, prune=prune)}

        results = {}
        for subdir in subdirs:
            source_type = "true" if "true" in subdir.lower() else "noisy" if "noisy" in subdir.lower() else subdir
            results[source_type] = process_directory(
                os.path.join(base_dir, subdir), source_type, force=force, prune=prune
            )
        return results


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA [--wipe] [--force] [--no-prune]
    #   python -m app.ingestion.processor DATA/true_data true
    flags = {"--wipe", "--force", "--no-prune"}
    clean_args = [a for a in sys.argv if a not in flags]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    summary = run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe="--wipe" in sys.argv,
        force="--force" in sys.argv,
        prune="--no-prune" not in sys.argv,
    )
    print(f"Ingestion summary: {summary}")
    logfire.info("Ingestion job completed.")
