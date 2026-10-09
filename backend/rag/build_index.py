# backend/rag/build_index.py
"""
Build the ChromaDB vector index from glossary.json.

Run manually:  python -m backend.rag.build_index
Or let it happen automatically on first app start (see rag/retriever.py).
"""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_glossary"
BATCH_SIZE = 100
GLOSSARY_PATH = Path("backend/rag/glossary.json")


def build_index() -> int:
    """
    (Re)build the medical glossary index. Returns the number of terms indexed.
    Raises FileNotFoundError with a clear message if glossary.json is missing.
    """
    if not GLOSSARY_PATH.exists():
        raise FileNotFoundError(
            f"{GLOSSARY_PATH} not found. Build it first:\n"
            f"  python scripts/scrape_invitro.py\n"
            f"  python scripts/scrape_wiki_ru.py\n"
            f"  python scripts/build_glossary.py"
        )

    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        terms = json.load(f)

    # Skip empty entries that would break ChromaDB
    terms = [t for t in terms if t.get("plain") and len(t["plain"].strip()) > 20]
    if not terms:
        raise ValueError("glossary.json contains no usable entries.")

    logger.info("Indexing %d terms from %s", len(terms), GLOSSARY_PATH)

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    for i in range(0, len(terms), BATCH_SIZE):
        batch = terms[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[t["term"] for t in batch],
            documents=[t["plain"] for t in batch],
            metadatas=[
                {
                    "term": t["term"],
                    "category": t.get("category", ""),
                    "unit": t.get("unit", ""),
                    "reference_range": t.get("reference_range", ""),
                }
                for t in batch
            ],
        )

    logger.info("✅ Indexed %d terms into '%s'", collection.count(), COLLECTION_NAME)
    return collection.count()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index()