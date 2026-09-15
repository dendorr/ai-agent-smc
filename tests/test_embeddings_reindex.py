"""Smoke tests for scripts/embeddings.py and scripts/reindex.py.

These tests do not call Ollama: they use an in-memory ChromaDB client and a
fake embedding function.
"""

import os
import sys
import uuid

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import chromadb  # noqa: E402
from chromadb.api.types import EmbeddingFunction  # noqa: E402

import embeddings  # noqa: E402
import reindex  # noqa: E402


class FakeEmbedding(EmbeddingFunction):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, input):
        return [np.full(self.dim, 0.1, dtype=np.float32) for _ in input]

    @staticmethod
    def name() -> str:
        return "fake"


def _collection(dim: int):
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"test-{uuid.uuid4().hex[:8]}",
        embedding_function=FakeEmbedding(dim),
    )


def test_stored_dimension_empty_collection():
    assert embeddings.stored_dimension(_collection(8)) is None


def test_verify_dimension_accepts_matching_vectors():
    col = _collection(8)
    col.upsert(ids=["a"], documents=["testo"])
    assert embeddings.stored_dimension(col) == 8
    embeddings.verify_collection_dimension(col, 8)


def test_verify_dimension_rejects_mismatch():
    col = _collection(8)
    col.upsert(ids=["a"], documents=["testo"])
    with pytest.raises(embeddings.EmbeddingDimensionMismatch):
        embeddings.verify_collection_dimension(col, 16)


def test_scan_sources_filters_extension_and_size(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "sub" / "b.PDF").write_bytes(b"x")
    (tmp_path / "c.exe").write_bytes(b"x")
    (tmp_path / "big.pdf").write_bytes(b"x" * (1024 * 1024 + 1))

    files, too_large = reindex.scan_sources(tmp_path, [".pdf"], max_file_mb=1)

    assert sorted(p.name for p in files) == ["a.pdf", "b.PDF"]
    assert [p.name for p in too_large] == ["big.pdf"]


def test_verify_collection_detects_count_mismatch_and_orphans():
    col = _collection(8)
    col.upsert(
        ids=["f1__c0", "f1__c1", "ghost__c0"],
        documents=["uno", "due", "tre"],
        metadatas=[
            {"path": "/data/f1.pdf", "type": "chunk"},
            {"path": "/data/f1.pdf", "type": "chunk"},
            {"path": "/data/ghost.pdf", "type": "chunk"},
        ],
    )
    results = [
        reindex.FileResult(path="/data/f1.pdf", agent="documents", mtime=0.0, size=1, chunks=3),
    ]
    report = reindex.AgentReport(agent="documents")

    reindex.verify_collection(report, col, results, expected_dim=8)

    assert not report.ok
    assert any("f1.pdf" in p for p in report.problems)
    assert any("orphan" in p for p in report.problems)


def test_verify_collection_passes_on_consistent_data():
    col = _collection(8)
    col.upsert(
        ids=["f1__c0", "f1__c1"],
        documents=["uno", "due"],
        metadatas=[
            {"path": "/data/f1.pdf", "type": "chunk"},
            {"path": "/data/f1.pdf", "type": "chunk"},
        ],
    )
    results = [
        reindex.FileResult(path="/data/f1.pdf", agent="documents", mtime=0.0, size=1, chunks=2),
    ]
    report = reindex.AgentReport(agent="documents")

    reindex.verify_collection(report, col, results, expected_dim=8)

    assert report.ok, report.problems
    assert report.stored_chunks == 2
