# backend/rag/retriever.py
"""
Retrieve plain-language medical term definitions from ChromaDB.

The collection is opened lazily on first use and built automatically
from glossary.json if it does not exist yet — so the app starts cleanly
in a fresh Docker container with an empty volume.
"""

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_glossary"

_client: chromadb.api.ClientAPI | None = None
_collection = None


def _get_client() -> chromadb.api.ClientAPI:
    """ChromaDB client with telemetry disabled (avoids posthog warnings)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    """
    Open the glossary collection, building it on demand if missing.
    Called lazily so a missing index never crashes the app at import time.
    """
    global _collection
    if _collection is not None:
        return _collection

    client = _get_client()
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    try:
        _collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )
        logger.info(
            "Loaded collection '%s' with %d terms.",
            COLLECTION_NAME,
            _collection.count(),
        )
    except Exception:
        logger.warning(
            "Collection '%s' not found at %s — building it from glossary.json.",
            COLLECTION_NAME,
            settings.chroma_persist_dir,
        )
        # Imported here to keep module import cheap and avoid cycles
        from backend.rag.build_index import build_index

        build_index()
        _collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )

    return _collection


def get_context(terms: list[str], top_k: int = 5) -> str:
    """
    Given medical terms extracted from a document, return their
    plain-language explanations formatted for injection into an LLM prompt.
    """
    if not terms:
        return ""

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        logger.warning("Collection is empty — returning no RAG context.")
        return ""

    results = collection.query(
        query_texts=terms,
        n_results=min(top_k, count),
    )

    context_parts: list[str] = []
    seen: set[str] = set()

    for docs, metas in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(docs, metas):
            term_name = meta.get("term", "")
            if term_name in seen:
                continue
            seen.add(term_name)

            entry = f"**{term_name}**: {doc}"
            if unit := meta.get("unit"):
                entry += f"\n  - Единицы измерения: {unit}"
            if ref := meta.get("reference_range"):
                entry += f"\n  - Референсные значения: {ref}"
            context_parts.append(entry)

    return "\n\n".join(context_parts)


def index_status() -> dict:
    """Used by /health/ready to report RAG state."""
    try:
        col = _get_collection()
        return {"collection": COLLECTION_NAME, "terms": col.count(), "ready": True}
    except Exception as e:
        return {"collection": COLLECTION_NAME, "terms": 0, "ready": False, "error": str(e)}