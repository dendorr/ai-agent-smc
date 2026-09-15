"""Shared embedding setup for all ChromaDB collections.

Every collection in the project must be opened through this module so that:

- the same explicit multilingual embedding model is used for indexing and
  querying (never the ChromaDB default all-MiniLM-L6-v2, which is English-only,
  384-dimensional and truncates input at 256 tokens);
- a dimension mismatch between the configured model and the vectors already
  stored in a collection fails loudly at startup instead of degrading retrieval
  silently.

Embeddings are served by an Ollama-compatible /api/embed endpoint configured via
EMBED_BASE_URL and EMBED_MODEL in config/config.py.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction  # noqa: E402

from config.config import (  # noqa: E402
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_TIMEOUT_SECONDS,
)

logger = logging.getLogger("embeddings")

# Collection metadata applied on creation. Cosine distance is the appropriate
# space for normalized sentence embeddings such as bge-m3 and nomic-embed-text.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

_PROBE_TEXT = "dimension probe"

_embedding_fn: OllamaEmbeddingFunction | None = None
_embedding_dim: int | None = None


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when stored vectors do not match the configured embedding model."""


def get_embedding_function() -> OllamaEmbeddingFunction:
    """Return the shared embedding function, creating it lazily."""
    global _embedding_fn

    if _embedding_fn is None:
        _embedding_fn = OllamaEmbeddingFunction(
            url=EMBED_BASE_URL,
            model_name=EMBED_MODEL,
            timeout=EMBED_TIMEOUT_SECONDS,
        )
        logger.info("Embedding function: %s @ %s", EMBED_MODEL, EMBED_BASE_URL)

    return _embedding_fn


def get_embedding_dimension() -> int:
    """Return the output dimension of the configured embedding model.

    The value is obtained with a single probe call and cached for the process
    lifetime. The call also verifies that the embedding backend is reachable
    and that the model is available.
    """
    global _embedding_dim

    if _embedding_dim is None:
        vectors = get_embedding_function()([_PROBE_TEXT])
        _embedding_dim = int(len(vectors[0]))
        logger.info("Embedding dimension for %s: %s", EMBED_MODEL, _embedding_dim)

    return _embedding_dim


def stored_dimension(collection: Any) -> int | None:
    """Return the dimension of vectors already stored in a collection.

    Returns None when the collection is empty.
    """
    if collection.count() == 0:
        return None

    sample = collection.peek(limit=1)
    embeddings = sample.get("embeddings")

    if embeddings is None or len(embeddings) == 0:
        return None

    return int(len(embeddings[0]))


def verify_collection_dimension(collection: Any, expected_dim: int) -> None:
    """Raise EmbeddingDimensionMismatch when stored vectors have another size."""
    actual = stored_dimension(collection)

    if actual is None or actual == expected_dim:
        return

    raise EmbeddingDimensionMismatch(
        f"Collection '{collection.name}' stores {actual}-dimensional vectors but "
        f"EMBED_MODEL={EMBED_MODEL} produces {expected_dim}-dimensional vectors. "
        "Delete the collection directory and the watcher registry, then re-index."
    )


def open_collection(client: Any, name: str, verify: bool = True) -> Any:
    """Open or create a collection bound to the shared embedding function.

    Args:
        client: A chromadb client (PersistentClient, HttpClient or EphemeralClient).
        name: Collection name.
        verify: When True, compare stored vector size with the configured model
            and raise EmbeddingDimensionMismatch on mismatch.
    """
    collection = client.get_or_create_collection(
        name,
        embedding_function=get_embedding_function(),
        metadata=COLLECTION_METADATA,
    )

    if verify:
        verify_collection_dimension(collection, get_embedding_dimension())

    return collection
