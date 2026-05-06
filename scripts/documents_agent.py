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
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CHROMA_PATHS,
    LLM_MODEL_FAST,
    LLM_MODEL_MAIN,
)

import chromadb
import semantic_analyzer as analyzer
from llm_client import chat_complete, chat_complete_json

client = chromadb.PersistentClient(path=CHROMA_PATHS["documents"])
collection = client.get_or_create_collection("documents")


# ── Models ────────────────────────────────────────────────────────────────────
ROUTING_MODEL = LLM_MODEL_FAST  # fast: selects relevant documents
ANSWER_MODEL = LLM_MODEL_MAIN  # accurate: final answer

# ── Document helper modules ─────────────────────────────────────────────────
from documents.memory import add_annotation, load_memory
from documents.indexing import index_file as _index_file, index_folder as _index_folder
from documents.filename_filter import detect_filename_filter as _detect_filename_filter


# ── Chunking + indexing (sync — called by the sync watcher) ───────────────────



# ── Indexing wrappers (sync — called by the sync watcher) ────────────────────

def index_file(filepath) -> int:
    """Index one document using the shared ChromaDB collection."""
    return _index_file(filepath, collection)


def index_folder(folder) -> None:
    """Index all supported document files from a folder recursively."""
    _index_folder(folder, collection)

# ── Filename detection ────────────────────────────────────────────────────────

def detect_filename_filter(query: str) -> dict | None:
    """Detect whether a query refers to a specific indexed filename."""
    return _detect_filename_filter(query, collection)

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