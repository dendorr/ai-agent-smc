"""Markdown cache helpers for the documents agent.

This module owns deterministic cache paths, cache freshness checks, and
read/write handling for Markdown generated from source documents.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from config.config import MARKDOWN_CACHE_DIR
from documents.converters import (
    convert_docx_to_markdown,
    convert_pdf_to_markdown,
    convert_pptx_to_markdown,
)

MARKDOWN_CACHE = Path(MARKDOWN_CACHE_DIR)


def _cache_path(filepath) -> Path:
    """Return the deterministic Markdown cache path for a source file."""
    h = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
    stem = re.sub(r"[^a-zA-Z0-9]", "_", Path(filepath).stem[:40]).strip("_")
    return MARKDOWN_CACHE / f"{stem}_{h}.md"


def _cache_is_valid(filepath, cache_file: Path) -> bool:
    """Return True when the cache exists and is newer than the source file."""
    if not cache_file.exists():
        return False

    try:
        return cache_file.stat().st_mtime >= Path(filepath).stat().st_mtime
    except Exception:
        return False


def get_or_create_markdown(filepath) -> str:
    """Return Markdown for a document, using a fresh cache when available.

    The cache is automatically invalidated when the source file modification
    time is newer than the cached Markdown file.
    """
    MARKDOWN_CACHE.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_path(filepath)
    if _cache_is_valid(filepath, cache_file):
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception:
            pass

    p = Path(filepath)
    ext = p.suffix.lower()
    markdown = ""

    if ext == ".pdf":
        markdown = convert_pdf_to_markdown(filepath)
    elif ext in (".pptx", ".ppt"):
        markdown = convert_pptx_to_markdown(filepath)
    elif ext in (".docx", ".doc"):
        markdown = convert_docx_to_markdown(filepath)
    elif ext == ".md":
        markdown = p.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".txt":
        content = p.read_text(encoding="utf-8", errors="ignore")
        markdown = f"# {p.stem}\n\n{content}"

    if markdown and len(markdown.strip()) > 50:
        try:
            cache_file.write_text(markdown, encoding="utf-8")
            print(
                f" [cache] Written {cache_file.name} ({len(markdown):,} chars)",
                flush=True,
            )
        except Exception as e:
            print(f" [cache warning] {e}", flush=True)

    return markdown
