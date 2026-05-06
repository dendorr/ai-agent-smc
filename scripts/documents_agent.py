"""
DOCUMENTS AGENT v5 — Universal document intelligence (async + GLM-OCR vision)

All extraction functions (PPTX/DOCX/PDF) remain synchronous because
they would not benefit from async (they are CPU-bound + local file I/O).
LLM and ChromaDB calls are async.

Supports: PDF (including scanned), PPTX (with image OCR), DOCX, Markdown, TXT.

Architecture:
  - Markdown cache: convert one-time, reuse on restart (mtime-based)
  - 3-tier image OCR with automatic fallback:
      Tier 0: GLM-OCR (multimodal vision via Ollama) — top accuracy
      Tier 1: pytesseract — fast, CPU, local fallback
      Tier 2: easyocr     — neural, final fallback
  - Scanned PDFs: automatic detection + full-page rasterization
                  + GLM-OCR (extracts text, tables and formulas as Markdown)
  - Multi-model: routing model (LLM_MODEL_FAST) selects documents,
                 answer model (LLM_MODEL_MAIN) generates the response
  - Persistent memory: document index + user annotations
  - Zero hardcoded: no assumptions about content/language/domain
  - Air-gapped: GLM-OCR runs via Ollama locally, no cloud
  - Multi-step retrieval: MSA-inspired iterative search for multi-hop queries
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CHROMA_PATHS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXTENSIONS,
    LLM_MODEL_FAST,
    LLM_MODEL_MAIN,
    MARKDOWN_CACHE_DIR,
    MEMORY_PATH,
)

import chromadb
import semantic_analyzer as analyzer
from llm_client import chat_complete, chat_complete_json

client = chromadb.PersistentClient(path=CHROMA_PATHS["documents"])
collection = client.get_or_create_collection("documents")

MEMORY_FILE = Path(MEMORY_PATH) / "documents_memory.json"
MARKDOWN_CACHE = Path(MARKDOWN_CACHE_DIR)

# ── Models ────────────────────────────────────────────────────────────────────
ROUTING_MODEL = LLM_MODEL_FAST  # fast: selects relevant documents
ANSWER_MODEL = LLM_MODEL_MAIN  # accurate: final answer

# ── Document conversion helpers ──────────────────────────────────────────────
from documents.converters import (
    convert_docx_to_markdown,
    convert_pdf_to_markdown,
    convert_pptx_to_markdown,
)


# ── Markdown cache ────────────────────────────────────────────────────────────

def _cache_path(filepath) -> Path:
    """Deterministic path in cache for a source file."""
    h = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
    stem = re.sub(r"[^a-zA-Z0-9]", "_", Path(filepath).stem[:40]).strip("_")
    return MARKDOWN_CACHE / f"{stem}_{h}.md"


def _cache_is_valid(filepath, cache_file: Path) -> bool:
    """Cache hit: file exists AND is newer (or equal) than source."""
    if not cache_file.exists():
        return False
    try:
        return cache_file.stat().st_mtime >= Path(filepath).stat().st_mtime
    except Exception:
        return False


def get_or_create_markdown(filepath) -> str:
    """
    Return the Markdown for the document.
    Reads from cache if valid; otherwise converts and saves to cache.
    Automatic invalidation: regenerates when the source file changes.
    """
    MARKDOWN_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(filepath)

    if _cache_is_valid(filepath, cache_file):
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception:
            pass

    ext = Path(filepath).suffix.lower()
    markdown = ""

    if ext == ".pdf":
        markdown = convert_pdf_to_markdown(filepath)
    elif ext in (".pptx", ".ppt"):
        markdown = convert_pptx_to_markdown(filepath)
    elif ext in (".docx", ".doc"):
        markdown = convert_docx_to_markdown(filepath)
    elif ext == ".md":
        markdown = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    elif ext == ".txt":
        content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        markdown = f"# {Path(filepath).stem}\n\n{content}"

    if markdown and len(markdown.strip()) > 50:
        try:
            cache_file.write_text(markdown, encoding="utf-8")
            print(
                f"  [cache] Written {cache_file.name} ({len(markdown):,} chars)",
                flush=True,
            )
        except Exception as e:
            print(f"  [cache warning] {e}", flush=True)

    return markdown


# ── Document memory ───────────────────────────────────────────────────────────

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"documents": {}, "annotations": {}}


def save_memory(memory: dict):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def update_document_memory(filepath, markdown: str):
    """Save per-document metadata after indexing."""
    from datetime import datetime
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


def add_annotation(filename: str, note: str):
    """Attach a user annotation to a document (persistent)."""
    memory = load_memory()
    if "annotations" not in memory:
        memory["annotations"] = {}
    if filename not in memory["annotations"]:
        memory["annotations"][filename] = []
    memory["annotations"][filename].append(note)
    save_memory(memory)


# ── Chunking + indexing (sync — called by the sync watcher) ───────────────────

def chunk_text(text: str) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks or [""]


def index_file(filepath) -> int:
    """
    Convert a document to Markdown (or use cache), split into chunks and
    save to ChromaDB. Returns the number of indexed chunks, 0 if skipped.
    Synchronous function — the watcher is sync and runs outside the event loop.
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

    # Remove stale chunks
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


def index_folder(folder) -> None:
    """Index all supported document files from a folder recursively."""
    files = [
        f
        for f in Path(folder).rglob("*")
        if f.is_file() and f.suffix.lower() in EXTENSIONS["documents"]
    ]

    print(f"[Documents] Found {len(files)} files...")
    total = 0

    for f in files:
        n = index_file(str(f))
        if n:
            print(f"  [OK] {f.name} → {n} chunks")
        else:
            print(f"  [--] {f.name} → skipped")
        total += n

    print(f"[Documents] Done — {total} total chunks")

# ── Filename detection ────────────────────────────────────────────────────────

def detect_filename_filter(query: str) -> dict | None:
    """
    Analyze the user query to detect references to specific files.
    If a match is found, returns a ChromaDB 'where' filter for filename.
    Otherwise returns None (normal semantic search).

    Handles variants like: "lez 13", "Lez13", "lezione 13", "file lez13",
    "prova itinere 1", "ProvaItinere1", "mock exam", etc.
    """
    # Get all indexed filenames
    try:
        all_meta = collection.get(include=["metadatas"])
        all_filenames = list({
            m["filename"] for m in all_meta["metadatas"]
            if m.get("filename") and m.get("type") != "semantic_card"
        })
    except Exception:
        return None

    if not all_filenames:
        return None

    # Normalize: remove extension, spaces, underscores, all lowercase
    def normalize(s: str) -> str:
        s = Path(s).stem if "." in s else s
        # Split camelCase/PascalCase: "ProvaItinere1" → "prova itinere 1"
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        # Split letters from numbers: "Lez13" → "Lez 13"
        s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
        s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
        return re.sub(r"[_\-\s]+", " ", s).strip().lower()

    query_norm = normalize(query)

    # Find the filename with the best match
    best_match = None
    best_score = 0

    for fname in all_filenames:
        fname_norm = normalize(fname)

        # Exact match of normalized part
        if fname_norm in query_norm or query_norm in fname_norm:
            score = len(fname_norm)
            if score > best_score:
                best_score = score
                best_match = fname
            continue

        # Partial match: all filename words present in query
        fname_words = fname_norm.split()
        query_words = query_norm.split()
        if len(fname_words) >= 1 and all(fw in query_words for fw in fname_words):
            score = len(fname_norm)
            if score > best_score:
                best_score = score
                best_match = fname

    if best_match:
        print(f"  [smart-filter] Query '{query}' → filter for '{best_match}'", flush=True)
        return {"filename": best_match}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH — Multi-step retrieval (MSA-inspired Memory Interleave)
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT CHANGED (v5 → v5.1):
#   - Old search() renamed to _raw_search() (single-round, unchanged logic)
#   - New search() is a thin wrapper that calls multi_step_search()
#   - If MULTI_STEP_ENABLED=False, it falls through to a single _raw_search()
#   - server.py is NOT affected — same interface (search/answer/answer_stream)
# ══════════════════════════════════════════════════════════════════════════════

async def _raw_search(query: str) -> str:
    """Single-round vector DB search with smart filename filtering + lazy semantic cards."""
    where_filter = detect_filename_filter(query)
    return await analyzer.search_with_cards(
        collection, query, "documents", n_results=6, where_filter=where_filter
    )


async def search(query: str) -> str:
    """
    Multi-step retrieval search (MSA-inspired Memory Interleave).

    Round 1: standard ChromaDB search with filename filtering
    Round 2+: if the fast LLM determines context is insufficient,
              generates a follow-up query and searches again.
    Contexts are merged and deduplicated across rounds.
    Falls back to single-round search if MULTI_STEP_ENABLED=False.
    """
    from multi_step_retrieval import multi_step_search

    return await multi_step_search(query, _raw_search, "documents")


# ── Routing model (async) ─────────────────────────────────────────────────────

async def route_documents(query: str, memory: dict) -> list:
    """
    Use the fast model to identify the most likely relevant documents.
    Returns a list of filenames. On error, empty list.
    """
    docs = memory.get("documents", {})
    if not docs:
        return []

    doc_lines = "\n".join(
        f"  - {name}  ({info.get('words', 0):,} parole | "
        f"{info.get('size_kb', 0)} KB | {info.get('ext', '')})"
        for name, info in list(docs.items())[:25]
    )

    system = "Sei un assistente che seleziona documenti rilevanti. Rispondi solo con nomi file, uno per riga."
    user = f"""DOCUMENTI DISPONIBILI:
{doc_lines}

DOMANDA: {query}

Elenca SOLO i nomi file più rilevanti (massimo 4), uno per riga.
Solo nomi file esatti, nessun altro testo."""

    try:
        raw = await chat_complete_json(
            model=ROUTING_MODEL, system=system, user=user, temperature=0.0
        )
        all_docs = set(docs.keys())
        relevant = [
            line.strip().strip("-").strip()
            for line in raw.strip().split("\n")
            if line.strip() and any(d in line for d in all_docs)
        ]
        return relevant[:4]
    except Exception:
        return []


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sei un esperto analista documentale con accesso completo ai documenti aziendali.

REGOLE ASSOLUTE:
- Rispondi SEMPRE e SOLO in italiano, qualunque sia la lingua del documento
- Usa ESCLUSIVAMENTE le informazioni presenti nei documenti forniti nel contesto
- Non inventare, non stimare, non integrare con conoscenze esterne
- Cita sempre la fonte (nome file, numero slide, numero pagina) per ogni dato riportato
- Se l'informazione non è presente nei documenti, dichiaralo esplicitamente

NUMERI, MISURE E SPECIFICHE TECNICHE — REGOLA CRITICA:
- Riporta i valori ESATTAMENTE come appaiono nel documento: nessuna trasformazione
- Preserva le unità di misura (mm, cm, m, µm, kg, g, ml, l, %, °C, bar, N, pz, €, $, ecc.)
- Non arrotondare mai, non convertire unità, non cambiare il formato
- Tolleranze (±0,5; ±5%; max 3 mm) vanno sempre riportate insieme al valore principale
- Tabelle con misure vanno riportate COMPLETE e FEDELI all'originale
- Se il documento riporta un range (es. 10-15 kg), riporta il range intero

TESTO ESTRATTO DA IMMAGINI ([OCR]):
- Il testo tra [OCR] proviene da immagini, grafici o schemi nel documento
- Trattalo come dato attendibile; segnala la fonte OCR se utile al contesto

STRUTTURA DELLE RISPOSTE:
- Specifiche tecniche → tabella o lista ordinata con unità
- Confronti tra documenti → colonne affiancate con fonte per ogni valore
- Riassunti → struttura gerarchica (sezioni → punti chiave)
- Ricerca di valori specifici → cita il contesto esatto del documento

CAPACITÀ:
- Analisi di presentazioni, relazioni, manuali, capitolati, specifiche tecniche
- Estrazione precisa di dati da tabelle, grafici e immagini (via OCR)
- Confronto tra più documenti sullo stesso argomento
- Ricerca di termini, misure, codici o specifiche esatte
- Sintesi di documenti lunghi mantenendo tutti i dati numerici"""


# ── Build user prompt — used by both answer() and streaming ───────────────────

async def _build_user_prompt(question: str, context: str) -> str:
    """
    Build the user prompt for the answer model, including:
      - index of indexed documents
      - any user annotations
      - hints from the routing model (probably relevant documents)
      - the search context (chunks + cards)
    """
    loop = asyncio.get_running_loop()
    memory = await loop.run_in_executor(None, load_memory)

    docs = memory.get("documents", {})
    mem_txt = ""

    if docs:
        doc_lines = "\n".join(
            f"  - {name} ({info.get('words', 0):,} parole | {info.get('ext', '')})"
            for name, info in list(docs.items())[:20]
        )
        mem_txt += f"\nDOCUMENTI INDICIZZATI:\n{doc_lines}\n"

    annotations = memory.get("annotations", {})
    if annotations:
        ann_lines = [
            f"  [{fname}] {note}"
            for fname, notes in list(annotations.items())[:5]
            for note in notes[:3]
        ]
        if ann_lines:
            mem_txt += "\nANNOTAZIONI UTENTE:\n" + "\n".join(ann_lines) + "\n"

    relevant = await route_documents(question, memory)
    if relevant:
        mem_txt += "\nDOCUMENTI PROBABILMENTE RILEVANTI PER QUESTA DOMANDA:\n"
        mem_txt += "\n".join(f"  - {d}" for d in relevant) + "\n"

    return f"""{mem_txt}
=== CONTENUTO DOCUMENTI — FONTE PRIMARIA ===
{context}

DOMANDA: {question}

RISPOSTA (obbligatoriamente in italiano — dati solo dalla FONTE PRIMARIA,
numeri e misure ESATTAMENTE come nel documento):"""


# ── Answer (async, non-streaming) ─────────────────────────────────────────────

async def answer(question: str, context: str) -> str:
    """Generate the non-streaming answer (called by server.py when stream=False)."""
    user_prompt = await _build_user_prompt(question, context)
    return await chat_complete(
        model=ANSWER_MODEL,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
    )


# ── Answer streaming (async iterator) ─────────────────────────────────────────

async def answer_stream(question: str, context: str):
    """
    Generate the streaming answer (token-by-token).
    Used by the server when stream=True to forward tokens via SSE
    directly to Open WebUI.
    """
    from llm_client import chat_complete_stream

    user_prompt = await _build_user_prompt(question, context)
    async for chunk in chat_complete_stream(
        model=ANSWER_MODEL,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
    ):
        yield chunk


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    index_folder(folder)