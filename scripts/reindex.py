"""Full and clean re-index of the ChromaDB collections.

Use this script after changing EMBED_MODEL, CHUNK_SIZE or CHUNK_OVERLAP, or
whenever a collection is suspected to contain vectors from another model.

Steps, per agent:
1. Refuse to run while server.py or watcher.py are alive, because ChromaDB
   persistent storage must not be written by two processes at once.
2. Probe the embedding backend to get the output dimension of EMBED_MODEL.
3. Delete the collection through the ChromaDB API (no manual rm -rf needed).
4. Import the agent module, which recreates the collection bound to the
   shared embedding function from embeddings.py.
5. Index every source file in the agent folder sequentially, using the same
   folder routing, extension filter and size limit as watcher.py.
6. Verify the result: per-file chunk counts stored in ChromaDB must match the
   counts returned by index_file, no chunk may belong to a path outside the
   source set, and stored vector size must match the embedding model.
7. Write memory/watcher_registry.json in the watcher format, so the watcher
   does not index everything again at its next start.

Usage (from scripts/):
    python reindex.py --dry-run
    python reindex.py --yes
    python reindex.py --yes --agents documents financial

Exit code is 0 when every verification passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb  # noqa: E402

from config.config import (  # noqa: E402
    CHROMA_PATHS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EXTENSIONS,
    FOLDERS,
    MEMORY_PATH,
)
from embeddings import get_embedding_dimension, stored_dimension  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reindex")
logging.getLogger("httpx").setLevel(logging.WARNING)

AGENT_MODULES = {
    "financial": "financial_agent",
    "drawings": "drawings_agent",
    "documents": "documents_agent",
}

REGISTRY_FILE = Path(MEMORY_PATH) / "watcher_registry.json"

# Same default and environment variable as watcher.py.
DEFAULT_MAX_FILE_MB = int(os.environ.get("WATCHER_MAX_FILE_MB", 200))

CONFLICTING_SCRIPTS = ("server.py", "watcher.py")


@dataclass
class FileResult:
    """Outcome of indexing a single source file."""

    path: str
    agent: str
    mtime: float
    size: int
    chunks: int = 0
    error: str = ""
    seconds: float = 0.0


@dataclass
class AgentReport:
    """Verification outcome for one agent collection."""

    agent: str
    source_files: int = 0
    skipped_large: list[str] = field(default_factory=list)
    indexed_files: int = 0
    empty_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    expected_chunks: int = 0
    stored_items: int = 0
    stored_chunks: int = 0
    stored_dim: int | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.failed_files


def find_conflicting_processes() -> list[str]:
    """Return command lines of running server.py or watcher.py processes."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []

    own_pid = os.getpid()
    found: list[str] = []

    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue

        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue

        args = [part for part in raw.decode(errors="ignore").split("\0") if part]
        if not args or "python" not in Path(args[0]).name:
            continue

        if any(Path(arg).name in CONFLICTING_SCRIPTS for arg in args[1:]):
            found.append(f"pid {entry.name}: {' '.join(args)}")

    return found


def scan_sources(folder: Path, extensions: list[str], max_file_mb: int) -> tuple[list[Path], list[Path]]:
    """Return (files to index, files skipped for size) for one agent folder.

    Mirrors watcher.scan_and_report: recursive scan, suffix filter, size limit.
    """
    if not folder.is_dir():
        return [], []

    allowed = {ext.lower() for ext in extensions}
    limit = max_file_mb * 1024 * 1024
    files: list[Path] = []
    too_large: list[Path] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue

        if path.stat().st_size > limit:
            too_large.append(path)
        else:
            files.append(path)

    return files, too_large


def delete_collection(agent: str) -> None:
    """Delete the agent collection if it exists."""
    client = chromadb.PersistentClient(path=CHROMA_PATHS[agent])
    names = {getattr(item, "name", item) for item in client.list_collections()}

    if agent in names:
        client.delete_collection(agent)
        logger.info("[%s] Collection deleted", agent)
    else:
        logger.info("[%s] Collection not present, nothing to delete", agent)


def index_sources(agent: str, module: Any, files: list[Path]) -> list[FileResult]:
    """Index files sequentially through the agent index_file function."""
    results: list[FileResult] = []

    for position, path in enumerate(files, start=1):
        stat = path.stat()
        result = FileResult(
            path=str(path),
            agent=agent,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )
        started = time.perf_counter()

        try:
            result.chunks = int(module.index_file(str(path)) or 0)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"

        result.seconds = time.perf_counter() - started
        results.append(result)

        if result.error:
            logger.error("[%s] %d/%d %s failed: %s", agent, position, len(files), path.name, result.error)
        else:
            logger.info(
                "[%s] %d/%d %s -> %d chunks (%.1fs)",
                agent,
                position,
                len(files),
                path.name,
                result.chunks,
                result.seconds,
            )

    return results


def verify_collection(
    report: AgentReport,
    collection: Any,
    results: list[FileResult],
    expected_dim: int,
) -> None:
    """Compare what index_file reported with what ChromaDB actually stores."""
    expected = {r.path: r.chunks for r in results if not r.error and r.chunks > 0}
    report.expected_chunks = sum(expected.values())

    stored = collection.get(include=["metadatas"])
    report.stored_items = len(stored["ids"])

    stored_per_path: dict[str, int] = {}
    non_chunk_items = 0

    for metadata in stored["metadatas"]:
        metadata = metadata or {}
        if metadata.get("type", "chunk") != "chunk":
            non_chunk_items += 1
            continue
        path = str(metadata.get("path", ""))
        stored_per_path[path] = stored_per_path.get(path, 0) + 1

    report.stored_chunks = sum(stored_per_path.values())

    if non_chunk_items:
        report.problems.append(f"{non_chunk_items} non-chunk items present right after re-index")

    if report.stored_chunks != report.expected_chunks:
        report.problems.append(
            f"stored chunks {report.stored_chunks} != chunks reported by index_file {report.expected_chunks}"
        )

    for path, count in expected.items():
        actual = stored_per_path.get(path, 0)
        if actual != count:
            report.problems.append(f"{Path(path).name}: stored {actual} chunks, expected {count}")

    for path in sorted(set(stored_per_path) - set(expected)):
        report.problems.append(f"orphan chunks for path not in source set: {path}")

    report.stored_dim = stored_dimension(collection)
    if report.stored_dim is not None and report.stored_dim != expected_dim:
        report.problems.append(f"stored vector dimension {report.stored_dim} != {expected_dim}")


def write_registry(results: list[FileResult], agents: list[str], full_run: bool) -> None:
    """Write the watcher registry, keeping entries of agents not re-indexed."""
    registry: dict[str, Any] = {}

    if REGISTRY_FILE.exists():
        try:
            previous = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}

        backup = REGISTRY_FILE.with_name(f"{REGISTRY_FILE.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        backup.write_text(json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Previous registry saved to %s", backup)

        if not full_run:
            reindexed_roots = [FOLDERS[agent].resolve() for agent in agents]
            registry = {
                key: value
                for key, value in previous.items()
                if not any(Path(key).resolve().is_relative_to(root) for root in reindexed_roots)
            }

    indexed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    for result in results:
        if result.error or result.chunks <= 0:
            continue
        registry[result.path] = {
            "mtime": result.mtime,
            "chunks": result.chunks,
            "indexed_at": indexed_at,
        }

    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REGISTRY_FILE)
    logger.info("Registry written: %d entries", len(registry))


def print_summary(reports: list[AgentReport], expected_dim: int, elapsed: float) -> None:
    """Print a compact human-readable summary."""
    print()
    print(f"Embedding model: {EMBED_MODEL} @ {EMBED_BASE_URL} ({expected_dim} dims)")
    print(f"Chunking: {CHUNK_SIZE} words, overlap {CHUNK_OVERLAP}")
    print(f"Elapsed: {elapsed:.1f}s")
    print()
    header = f"{'agent':<10} {'sources':>7} {'indexed':>7} {'empty':>5} {'failed':>6} {'chunks':>6} {'stored':>6} {'dim':>5}  status"
    print(header)
    print("-" * len(header))

    for r in reports:
        print(
            f"{r.agent:<10} {r.source_files:>7} {r.indexed_files:>7} {len(r.empty_files):>5} "
            f"{len(r.failed_files):>6} {r.expected_chunks:>6} {r.stored_chunks:>6} "
            f"{str(r.stored_dim or '-'):>5}  {'OK' if r.ok else 'FAIL'}"
        )

    for r in reports:
        for name in r.skipped_large:
            print(f"  [{r.agent}] skipped, over size limit: {name}")
        for name in r.empty_files:
            print(f"  [{r.agent}] no text extracted, not indexed: {name}")
        for name in r.failed_files:
            print(f"  [{r.agent}] failed: {name}")
        for problem in r.problems:
            print(f"  [{r.agent}] PROBLEM: {problem}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full clean re-index of ChromaDB collections.")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=sorted(AGENT_MODULES),
        default=sorted(AGENT_MODULES),
        help="Agents to re-index (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan sources only, change nothing.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=DEFAULT_MAX_FILE_MB,
        help="Skip files larger than this size, same as the watcher.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    agents: list[str] = list(args.agents)
    full_run = set(agents) == set(AGENT_MODULES)

    sources: dict[str, tuple[list[Path], list[Path]]] = {
        agent: scan_sources(FOLDERS[agent], EXTENSIONS[agent], args.max_file_mb) for agent in agents
    }

    for agent in agents:
        files, too_large = sources[agent]
        logger.info(
            "[%s] %s -> %d files to index, %d over size limit",
            agent,
            FOLDERS[agent],
            len(files),
            len(too_large),
        )

    if args.dry_run:
        for agent in agents:
            for path in sources[agent][0]:
                print(f"  [{agent}] {path.relative_to(FOLDERS[agent])}")
        return 0

    conflicts = find_conflicting_processes()
    if conflicts:
        logger.error("Stop server.py and watcher.py before re-indexing:")
        for line in conflicts:
            logger.error("  %s", line)
        return 1

    try:
        expected_dim = get_embedding_dimension()
    except Exception as exc:
        logger.error("Embedding backend unavailable (%s @ %s): %s", EMBED_MODEL, EMBED_BASE_URL, exc)
        logger.error("Check that Ollama is running and run: ollama pull %s", EMBED_MODEL)
        return 1

    if not args.yes:
        answer = input(f"Delete and rebuild collections {agents} with {EMBED_MODEL}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes", "s", "si"}:
            logger.info("Aborted.")
            return 1

    started = time.perf_counter()
    all_results: list[FileResult] = []
    reports: list[AgentReport] = []

    for agent in agents:
        files, too_large = sources[agent]
        report = AgentReport(
            agent=agent,
            source_files=len(files),
            skipped_large=[p.name for p in too_large],
        )

        delete_collection(agent)
        module = importlib.import_module(AGENT_MODULES[agent])

        results = index_sources(agent, module, files)
        all_results.extend(results)

        report.indexed_files = sum(1 for r in results if not r.error and r.chunks > 0)
        report.empty_files = [Path(r.path).name for r in results if not r.error and r.chunks <= 0]
        report.failed_files = [Path(r.path).name for r in results if r.error]

        verify_collection(report, module.collection, results, expected_dim)
        reports.append(report)

    write_registry(all_results, agents, full_run)
    print_summary(reports, expected_dim, time.perf_counter() - started)

    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
