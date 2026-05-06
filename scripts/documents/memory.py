"""Persistent document memory helpers for the documents agent.

This module owns the documents_memory.json file and keeps document metadata
and user annotations out of the main agent module.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.config import MEMORY_PATH

MEMORY_FILE = Path(MEMORY_PATH) / "documents_memory.json"


def load_memory() -> dict:
    """Load persistent document memory from disk."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {"documents": {}, "annotations": {}}


def save_memory(memory: dict) -> None:
    """Persist document memory to disk."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def update_document_memory(filepath, markdown: str) -> None:
    """Save per-document metadata after indexing."""
    memory = load_memory()
    if "documents" not in memory:
        memory["documents"] = {}

    p = Path(filepath)
    memory["documents"][p.name] = {
        "filepath": str(filepath),
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        "words": len(markdown.split()),
        "chars": len(markdown),
        "size_kb": round(p.stat().st_size / 1024, 1),
        "ext": p.suffix.lower(),
    }

    save_memory(memory)


def add_annotation(filename: str, note: str) -> None:
    """Attach a user annotation to a document."""
    memory = load_memory()
    if "annotations" not in memory:
        memory["annotations"] = {}

    if filename not in memory["annotations"]:
        memory["annotations"][filename] = []

    memory["annotations"][filename].append(note)
    save_memory(memory)
