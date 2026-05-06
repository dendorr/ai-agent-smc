"""
SEMANTIC ANALYZER — Lazy evaluation, full async

Semantic cards are generated ONLY on first query (not during indexing,
to avoid slowing down the watcher). Once generated they are cached
permanently (on disk + inside ChromaDB).

All LLM calls are async via llm_client.AsyncOpenAI.
ChromaDB is synchronous by nature → wrapped in run_in_executor.
"""

import sys
import os
import json
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LLM_MODEL_FAST, MEMORY_PATH

from llm_client import chat_complete

MEMORY_FILE = Path(MEMORY_PATH) / "semantic_cards.json"


# ── Card persistence on disk ─────────────────────────────────────────────────

def load_cards() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cards(cards: dict):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Semantic card prompts per agent type ──────────────────────────────────────

_SYSTEM_PROMPT = (
    "Sei un assistente che produce schede semantiche strutturate "
    "per descrivere documenti aziendali. Rispondi in italiano, "
    "compilando ESATTAMENTE i campi richiesti, uno per riga."
)


def _user_prompt(filename: str, raw_text: str, agent_type: str) -> str:
    """Build the user prompt specific to the agent type."""
    preview = raw_text[:3000]

    if agent_type == "financial":
        fields = """TIPO_DOCUMENTO: [foglio spese / fattura / bilancio / preventivo / lista pagamenti / report di mercato]
CONTESTO_BUSINESS: [descrivi in 1-2 frasi cosa rappresenta questo file]
ENTITA_PRINCIPALI: [persone, aziende, prodotti menzionati]
VALORI_CHIAVE: [importi, totali, prezzi unitari importanti]
PERIODO: [anno, mese, data se presente]
STRUTTURA: [descrivi le colonne/campi principali]
COLORI_EXCEL: [spiega cosa significano i colori in questo file specifico, se presenti]
REGOLE_CALCOLO: [regole inferite dal contenuto, se evidenti]
NOTE_IMPORTANTI: [qualsiasi cosa utile per rispondere a domande su questo file]"""

    elif agent_type == "drawings":
        fields = """TIPO_FILE: [DXF/STP/IFC/SVG/STL/PDF tecnico]
TIPO_OGGETTO: [componente meccanico / edificio / struttura / assemblaggio]
FUNZIONE_PROBABILE: [a cosa potrebbe servire basandoti sulla geometria]
GEOMETRIA: [numero solidi, facce, superfici - complessità]
MATERIALI: [se presenti]
UNITA_MISURA: [mm/cm/m/pollici]
LAYER_PRINCIPALI: [se DXF/SVG]
ELEMENTI_BIM: [se IFC: piani, pareti, porte, ecc.]
NOTE_TECNICHE: [qualsiasi dettaglio tecnico rilevante]"""

    else:  # documents (default)
        fields = """TIPO_DOCUMENTO: [relazione/presentazione/manuale/contratto/procedura]
ARGOMENTO_PRINCIPALE: [di cosa tratta in 1-2 frasi]
STRUTTURA: [capitoli/sezioni/slide principali]
ENTITA_CHIAVE: [persone, aziende, prodotti, luoghi]
DATE_IMPORTANTI: [date rilevanti se presenti]
PUNTI_CHIAVE: [3-5 informazioni più importanti]"""

    return (
        f"FILE: {filename}\n"
        f"CONTENUTO ESTRATTO:\n{preview}\n\n"
        f"Produci una scheda semantica con ESATTAMENTE questo formato:\n\n"
        f"{fields}"
    )


# ── Card generation (async) ──────────────────────────────────────────────────

async def generate_semantic_card(filename: str, raw_text: str, agent_type: str) -> str:
    """Ask the fast model to analyze the file and produce a semantic card."""
    user_prompt = _user_prompt(filename, raw_text, agent_type)

    card = await chat_complete(
        model=LLM_MODEL_FAST,
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.1,
        timeout=60,
    )

    return f"=== SCHEDA SEMANTICA: {filename} ===\n{card}\n"


# ── Get-or-create card with ChromaDB + disk cache ────────────────────────────

async def get_or_create_card(filepath: str, raw_text: str,
                              agent_type: str, collection) -> str:
    """
    Retrieve the card from cache (ChromaDB) or generate it now.
    All ChromaDB operations (synchronous) are delegated to an executor.
    """
    filename = Path(filepath).name
    card_id  = f"{filepath}__semantic_card"
    loop     = asyncio.get_running_loop()

    # ── Check ChromaDB cache ─────────────────────────────────────────────────
    try:
        existing = await loop.run_in_executor(
            None, lambda: collection.get(ids=[card_id])
        )
        if existing and existing.get("ids") and existing.get("documents"):
            return existing["documents"][0]
    except Exception:
        pass

    # ── Generation (lazy, first time) ────────────────────────────────────────
    print(f"  [AI] Generating semantic card for {filename}...", flush=True)
    card = await generate_semantic_card(filename, raw_text, agent_type)

    # ── Save to ChromaDB (sync → executor) ───────────────────────────────────
    try:
        await loop.run_in_executor(
            None,
            lambda: collection.upsert(
                documents=[card],
                ids=[card_id],
                metadatas=[{
                    "filename": filename,
                    "path":     str(filepath),
                    "chunk":    -1,
                    "agent":    agent_type,
                    "type":     "semantic_card",
                }],
            ),
        )
    except Exception as e:
        print(f"  [warn] save card → ChromaDB: {e}", flush=True)

    # ── Save to disk (JSON file) — also via executor ─────────────────────────
    try:
        await loop.run_in_executor(None, _persist_card_to_disk, str(filepath), card)
    except Exception as e:
        print(f"  [warn] save card → disk: {e}", flush=True)

    return card


def _persist_card_to_disk(filepath: str, card: str):
    """Sync helper to write the card to disk (called from executor)."""
    cards = load_cards()
    cards[filepath] = card
    save_cards(cards)


# ── Search with lazy card generation ─────────────────────────────────────────

async def search_with_cards(collection, query: str, agent_type: str,
                             n_results: int = 6, where_filter: dict = None) -> str:
    """
    Run similarity search on ChromaDB and enrich context with semantic
    cards for the relevant files (generated on-the-fly if missing).

    Accepts an optional where_filter dict to filter by metadata
    (e.g. {"filename": "Lez13.pdf"}) for direct filename matching.

    All ChromaDB operations run in executor.
    """
    loop = asyncio.get_running_loop()

    count = await loop.run_in_executor(None, collection.count)
    if count == 0:
        return "[No documents indexed yet]"

    n = min(n_results, count)
    if where_filter:
        r = await loop.run_in_executor(
            None,
            lambda: collection.query(query_texts=[query], n_results=n, where=where_filter),
        )
    else:
        r = await loop.run_in_executor(
            None,
            lambda: collection.query(query_texts=[query], n_results=n),
        )

    cards = []
    chunks = []
    seen_files = set()

    # Separate semantic cards from raw chunks
    for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
        fname = meta.get("filename", "")
        chunk_type = meta.get("type", "chunk")
        if chunk_type == "semantic_card":
            if fname not in seen_files:
                cards.append((doc, meta))
                seen_files.add(fname)
        else:
            chunks.append((doc, meta))

    # For each file in results, get the card (generate if missing) — in parallel
    async def _fetch_card_for(meta):
        fname = meta.get("filename", "")
        fpath = meta.get("path", "")
        if fname in seen_files or not fpath:
            return None

        card_id = f"{fpath}__semantic_card"
        try:
            existing = await loop.run_in_executor(
                None, lambda: collection.get(ids=[card_id])
            )
            if existing and existing.get("ids") and existing.get("documents"):
                return (existing["documents"][0], {"filename": fname})

            # Card missing → generate now (lazy)
            raw_chunks = await loop.run_in_executor(
                None, lambda: collection.get(where={"path": fpath})
            )
            if raw_chunks and raw_chunks.get("documents"):
                raw_text = " ".join(raw_chunks["documents"][:3])
                card = await get_or_create_card(fpath, raw_text, agent_type, collection)
                return (card, {"filename": fname})
        except Exception:
            return None
        return None

    # Launch at most one missing-card request per file.
    # Without this, multiple chunks from the same file can trigger parallel
    # duplicate semantic-card generations before the first one is saved.
    pending_files = {}

    for _, meta in chunks:
        fname = meta.get("filename", "")
        fpath = meta.get("path", "")

        if not fname or not fpath:
            continue

        if fname in seen_files:
            continue

        if fpath not in pending_files:
            pending_files[fpath] = meta

    tasks = [_fetch_card_for(meta) for meta in pending_files.values()]
    
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple):
                doc, meta = res
                if meta["filename"] not in seen_files:
                    cards.append(res)
                    seen_files.add(meta["filename"])

    # Build context: cards first, then raw chunks
    context = ""
    if cards:
        context += "=== DOCUMENT UNDERSTANDING ===\n"
        for doc, _ in cards:
            context += doc + "\n"
        context += "\n=== RAW DATA ===\n"
    for doc, meta in chunks:
        context += f"\n--- {meta.get('filename', 'unknown')} ---\n{doc[:2000]}\n"

    return context if context.strip() else "[No relevant content found]"