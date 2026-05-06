"""Text chunking helpers for the documents agent."""

from __future__ import annotations

from config.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks, i = [], 0

    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks or [""]
