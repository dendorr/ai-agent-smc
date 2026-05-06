"""Filename filter detection helpers for document retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _normalize_filename_reference(value: str) -> str:
    """Normalize a filename or query fragment for fuzzy filename matching."""
    value = Path(value).stem if "." in value else value

    # Split camelCase/PascalCase: "ProvaItinere1" -> "prova itinere 1".
    value = re.sub(r"([a-z])([A-Z])", r" ", value)

    # Split letters from numbers: "Lez13" -> "Lez 13".
    value = re.sub(r"([a-zA-Z])(\d)", r" ", value)
    value = re.sub(r"(\d)([a-zA-Z])", r" ", value)

    return re.sub(r"[_\-\s]+", " ", value).strip().lower()


def detect_filename_filter(query: str, collection: Any) -> dict | None:
    """
    Detect references to specific indexed files in the user query.

    Returns a ChromaDB where filter when a filename match is found, otherwise
    returns None so semantic search can proceed normally.
    """
    try:
        all_meta = collection.get(include=["metadatas"])
        all_filenames = list(
            {
                metadata["filename"]
                for metadata in all_meta["metadatas"]
                if metadata.get("filename") and metadata.get("type") != "semantic_card"
            }
        )
    except Exception:
        return None

    if not all_filenames:
        return None

    query_norm = _normalize_filename_reference(query)
    query_words = query_norm.split()

    best_match = None
    best_score = 0

    for filename in all_filenames:
        filename_norm = _normalize_filename_reference(filename)

        # Exact match of normalized part.
        if filename_norm in query_norm or query_norm in filename_norm:
            score = len(filename_norm)
            if score > best_score:
                best_score = score
                best_match = filename
            continue

        # Partial match: all filename words are present in the query.
        filename_words = filename_norm.split()
        if filename_words and all(word in query_words for word in filename_words):
            score = len(filename_norm)
            if score > best_score:
                best_score = score
                best_match = filename

    if best_match:
        print(
            f"  [smart-filter] Query '{query}' -> filter for '{best_match}'",
            flush=True,
        )
        return {"filename": best_match}

    return None
