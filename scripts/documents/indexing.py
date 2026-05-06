"""Document indexing helpers.

This module keeps document indexing separate from retrieval and answer generation.
The ChromaDB collection is passed in by the caller so the main documents agent
remains the single owner of the persistent client setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.config import EXTENSIONS
from documents.chunking import chunk_text
from documents.markdown_cache import get_or_create_markdown
from documents.memory import update_document_memory


def index_file(filepath, collection: Any) -> int:
    """
    Convert a document to Markdown (or use cache), split it into chunks and
    save it to ChromaDB. Returns the number of indexed chunks, or 0 if skipped.
    """
    p = Path(filepath)
    ext = p.suffix.lower()

    if ext not in EXTENSIONS["documents"]:
        return 0

    print(f"  [docs] Processing {p.name}...", flush=True)
    markdown = get_or_create_markdown(filepath)

    if not markdown or not markdown.strip():
        return 0

    update_document_memory(filepath, markdown)

    # Remove stale chunks.
    try:
        existing = collection.get(where={"path": str(filepath)})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    chunks = chunk_text(markdown)
    for i, chunk in enumerate(chunks):
        collection.upsert(
            documents=[chunk],
            ids=[f"{filepath}__c{i}"],
            metadatas=[
                {
                    "filename": p.name,
                    "path": str(filepath),
                    "chunk": i,
                    "agent": "documents",
                    "type": "chunk",
                    "ext": ext,
                }
            ],
        )

    return len(chunks)


def index_folder(folder, collection: Any) -> None:
    """Index all supported document files from a folder recursively."""
    files = [
        f
        for f in Path(folder).rglob("*")
        if f.is_file() and f.suffix.lower() in EXTENSIONS["documents"]
    ]

    print(f"[Documents] Found {len(files)} files...")
    total = 0

    for f in files:
        n = index_file(str(f), collection)
        if n:
            print(f"  [OK] {f.name} -> {n} chunks")
        else:
            print(f"  [--] {f.name} -> skipped")
        total += n

    print(f"[Documents] Done - {total} total chunks")
