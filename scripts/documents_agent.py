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
    OCR_ENABLED,
    OCR_MIN_TEXT_LEN,
    OCR_PDF_MIN_TEXT_PER_PAGE,
    OCR_PDF_PAGE_RASTER_DPI,
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

# ── OCR helpers ───────────────────────────────────────────────────────────────
from documents.ocr import ocr_image_bytes, ocr_image_file, ocr_with_glm


# ── Shared helpers ────────────────────────────────────────────────────────────

def _table_to_markdown(table) -> str:
    """
    Convert a Table object (python-pptx or python-docx) to Markdown table.
    Preserves all cell content exactly — no rounding.
    """
    rows = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(rows)


# ── PPTX → Markdown ───────────────────────────────────────────────────────────

def convert_pptx_to_markdown(filepath) -> str:
    """
    Convert a PPTX to structured Markdown.

    Per slide:
      - Number + title as heading
      - Text boxes (level-aware indentation)
      - Tables → Markdown (exact numbers/units)
      - Embedded images → OCR
      - Speaker notes → blockquote at end of slide
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    p = Path(filepath)
    prs = Presentation(str(filepath))

    lines = [f"# {p.stem}", f"*File: {p.name} — {len(prs.slides)} slide*", ""]

    for slide_num, slide in enumerate(prs.slides, 1):
        title = ""
        try:
            if slide.shapes.title and slide.shapes.title.text.strip():
                title = slide.shapes.title.text.strip()
        except Exception:
            pass

        slide_heading = f"## Slide {slide_num}"
        if title:
            slide_heading += f": {title}"
        lines.append(slide_heading)
        lines.append("")

        for shape in slide.shapes:
            if shape.has_table:
                lines.append(_table_to_markdown(shape.table))
                lines.append("")
                continue

            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if not text or text == title:
                        continue
                    level = getattr(para, "level", 0)
                    indent = "  " * level
                    prefix = f"{'#' * (level + 3)} " if level > 0 else ""
                    lines.append(f"{indent}{prefix}{text}")
                lines.append("")
                continue

            try:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    img_bytes = shape.image.blob
                    hint = f"[Immagine — slide {slide_num}]"
                    ocr_text = ocr_image_bytes(img_bytes, source_hint=hint)
                    if ocr_text:
                        quoted = ocr_text.replace("\n", "\n> ")
                        lines.append(f"> **{hint}**")
                        lines.append(f"> {quoted}")
                        lines.append("")
            except Exception:
                pass

        try:
            notes_tf = slide.notes_slide.notes_text_frame
            notes_text = notes_tf.text.strip()
            if notes_text:
                quoted = notes_text.replace("\n", "\n> ")
                lines.append("**Note relatore:**")
                lines.append(f"> {quoted}")
                lines.append("")
        except Exception:
            pass

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── DOCX → Markdown ───────────────────────────────────────────────────────────

def convert_docx_to_markdown(filepath) -> str:
    """
    Convert a DOCX to structured Markdown.

    Strategy:
      Tier 1 — markitdown (Microsoft): perfect headings/tables/lists
      Tier 2 — python-docx manual: iterates body in document order
               (paragraphs + tables interleaved), then extracts OCR images

    Numbers, units, tolerances always preserved exactly.
    """
    p = Path(filepath)

    # Tier 1: markitdown
    try:
        from markitdown import MarkItDown
        result = MarkItDown().convert(str(filepath))
        text = result.text_content.strip()
        if len(text) > 100:
            md = f"# {p.stem}\n*File: {p.name}*\n\n{text}"
            img_blocks = _docx_extract_image_ocr(filepath)
            if img_blocks:
                md += "\n\n## Immagini estratte dal documento\n\n" + "\n\n".join(img_blocks)
            return md
    except ImportError:
        pass
    except Exception as e:
        print(f"  [markitdown warning] {p.name}: {e}", flush=True)

    # Tier 2: python-docx manual
    try:
        from docx import Document
        from docx.text.paragraph import Paragraph
        from docx.table import Table as DocxTable

        doc = Document(str(filepath))
        lines = [f"# {p.stem}", f"*File: {p.name}*", ""]

        _HEADING_MAP = {
            "heading 1": "#",
            "heading 2": "##",
            "heading 3": "###",
            "heading 4": "####",
            "heading 5": "#####",
        }

        for child in doc.element.body.iterchildren():
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "p":
                para = Paragraph(child, doc)
                text = para.text.strip()
                style = (para.style.name or "").lower() if para.style else ""

                if not text:
                    lines.append("")
                    continue

                prefix = ""
                for key, md_prefix in _HEADING_MAP.items():
                    if key in style:
                        prefix = md_prefix
                        break

                if prefix:
                    lines.append(f"{prefix} {text}")
                elif "list" in style:
                    level_match = re.search(r"\d", style)
                    indent_level = int(level_match.group()) - 1 if level_match else 0
                    indent = "  " * indent_level
                    lines.append(f"{indent}- {text}")
                else:
                    lines.append(text)

            elif tag == "tbl":
                tbl = DocxTable(child, doc)
                lines.append("")
                lines.append(_table_to_markdown(tbl))
                lines.append("")

        lines.append("")

        img_blocks = _docx_extract_image_ocr(filepath)
        if img_blocks:
            lines.append("## Immagini estratte dal documento")
            lines.extend(img_blocks)

        return "\n".join(lines)

    except Exception as e:
        return f"[DOCX error: {e}]"


def _docx_extract_image_ocr(filepath) -> list:
    """Extract all embedded images from DOCX and run OCR on each."""
    blocks = []
    try:
        from docx import Document
        doc = Document(str(filepath))
        for i, rel in enumerate(doc.part.rels.values(), 1):
            if "image" in rel.reltype:
                try:
                    img_bytes = rel.target_part.blob
                    ocr_text = ocr_image_bytes(
                        img_bytes, source_hint=f"[Immagine {i}]"
                    )
                    if ocr_text and len(ocr_text) > 10:
                        quoted = ocr_text.replace("\n", "\n> ")
                        blocks.append(f"> **Immagine {i}:**\n> {quoted}")
                except Exception:
                    pass
    except Exception:
        pass
    return blocks


# ── PDF → Markdown ────────────────────────────────────────────────────────────

def convert_pdf_to_markdown(filepath) -> str:
    """
    Convert a PDF to structured Markdown.

    Tier 1 — markitdown:  best layout preservation on native PDFs
    Tier 2 — fitz:        text + OCR scanned pages (GLM-OCR)
                          + OCR embedded images + page-by-page tables
    Tier 3 — pdfplumber:  supplementary tables

    Scanned PDFs: pages with < OCR_PDF_MIN_TEXT_PER_PAGE chars are
    rasterized at OCR_PDF_PAGE_RASTER_DPI DPI and sent whole to GLM-OCR.

    All numerical values, units and tolerances preserved exactly.
    """
    import fitz as _fitz

    p = Path(filepath)
    header = f"# {p.stem}\n*File: {p.name}*\n\n"

    # Tier 1: markitdown
    try:
        from markitdown import MarkItDown
        result = MarkItDown().convert(str(filepath))
        text = result.text_content.strip()
        if len(text) > 100:
            return header + text
    except ImportError:
        pass
    except Exception as e:
        print(f"  [markitdown warning] {p.name}: {e}", flush=True)

    # Tier 2: fitz
    fitz_parts = []
    try:
        doc = _fitz.open(str(filepath))
        for page_num, page in enumerate(doc, 1):
            page_lines = [f"## Pagina {page_num}", ""]

            text = page.get_text("text")
            text_clean = text.strip()
            page_was_ocr = False  # True = page handled as scan (GLM-OCR)

            # Page with native text: use it
            if len(text_clean) >= OCR_PDF_MIN_TEXT_PER_PAGE:
                page_lines.append(text_clean)
                page_lines.append("")

            # Page with little text + OCR enabled: probable scan
            elif OCR_ENABLED:
                try:
                    # Full-page rasterization (PDF base = 72 DPI)
                    zoom = OCR_PDF_PAGE_RASTER_DPI / 72
                    mat = _fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    page_png = pix.tobytes("png")

                    ocr_text = ocr_with_glm(
                        page_png,
                        source_hint=f"[Pagina {page_num} (scansione)]",
                    )
                    if ocr_text and len(ocr_text) > OCR_MIN_TEXT_LEN:
                        page_lines.append(f"[OCR pagina {page_num} — scansione]")
                        page_lines.append(ocr_text)
                        page_lines.append("")
                        page_was_ocr = True
                    elif text_clean:
                        # GLM-OCR failed: use the little native text as fallback
                        page_lines.append(text_clean)
                        page_lines.append("")
                except Exception as e:
                    print(f"  [GLM-OCR warning] pag. {page_num}: {e}", flush=True)
                    if text_clean:
                        page_lines.append(text_clean)
                        page_lines.append("")

            # OCR disabled: use the little native text if present
            elif text_clean:
                page_lines.append(text_clean)
                page_lines.append("")

            # Tables via fitz (PyMuPDF >= 1.23)
            try:
                for tab in page.find_tables().tables:
                    df = tab.to_pandas()
                    md = df.to_markdown(index=False)
                    if md:
                        page_lines.append(f"**Tabella:**\n{md}")
                        page_lines.append("")
            except Exception:
                pass

            # Embedded images → OCR
            # Skip if the page was already handled via GLM-OCR rasterization:
            # the embedded images are already included in the rasterized output —
            # avoids duplicates in ChromaDB.
            if not page_was_ocr:
                for img_info in page.get_images(full=True):
                    try:
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img.get("image", b"")
                        ocr_text = ocr_image_bytes(
                            img_bytes, source_hint=f"[Immagine pag. {page_num}]"
                        )
                        if ocr_text and len(ocr_text) > OCR_MIN_TEXT_LEN:
                            quoted = ocr_text.replace("\n", "\n> ")
                            page_lines.append(f"> **Immagine pag. {page_num}:**\n> {quoted}")
                            page_lines.append("")
                    except Exception:
                        pass

            fitz_parts.append("\n".join(page_lines))
        doc.close()
    except Exception as e:
        fitz_parts.append(f"[fitz error: {e}]")

    # Tier 3: pdfplumber
    plumber_parts = []
    try:
        import pdfplumber
        with pdfplumber.open(str(filepath)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for table in page.extract_tables() or []:
                    rows = []
                    for i, row in enumerate(table):
                        cells = [str(c).strip() if c else "" for c in row]
                        rows.append("| " + " | ".join(cells) + " |")
                        if i == 0:
                            rows.append("|" + "|".join(["---"] * len(cells)) + "|")
                    if rows:
                        plumber_parts.append(
                            f"**Tabella pag. {page_num}:**\n" + "\n".join(rows)
                        )
    except Exception:
        pass

    result = header
    if fitz_parts:
        result += "\n\n---\n\n".join(fitz_parts)
    if plumber_parts:
        result += "\n\n## Tabelle aggiuntive\n\n" + "\n\n".join(plumber_parts)

    return result if len(result) > len(header) + 20 else f"[Impossibile estrarre {p.name}]"


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