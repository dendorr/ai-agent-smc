# ============================================================
# GLOBAL CONFIGURATION — Company AI Agent System
# ============================================================
# Tutto è configurabile via variabili d'ambiente (file .env oppure
# `export` in shell). I default qui sotto sono sensati per dev su
# WSL2/Ubuntu con Ollama locale e si trasformano automaticamente
# in produzione (vLLM/SGLang) cambiando solo LLM_BASE_URL.
# Nessun path Windows hardcoded — usare COMPANY_DATA_DIR.
# ============================================================

import os
from pathlib import Path

# ── Helper ────────────────────────────────────────────────────────────────────

def _env(name: str, default: str) -> str:
    """Legge una variabile d'ambiente; se vuota o non impostata, usa il default."""
    val = os.environ.get(name, "")
    return val if val else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


# ── Paths progetto ────────────────────────────────────────────────────────────

HOME_DIR = Path(os.path.expanduser("~"))
BASE_DIR = Path(_env("AI_AGENT_BASE_DIR", str(HOME_DIR / "ai-agent")))

# Cartella dati aziendali — file da indicizzare.
# Default: ~/ai-agent/data (con sotto-cartelle per agente).
# In produzione: puntare a un mount di rete (es: /mnt/company_server/...).
COMPANY_DATA_DIR = Path(_env("COMPANY_DATA_DIR", str(BASE_DIR / "data")))

FOLDERS = {
    "financial": Path(_env("FINANCIAL_DATA_DIR", str(COMPANY_DATA_DIR / "financial"))),
    "drawings":  Path(_env("DRAWINGS_DATA_DIR",  str(COMPANY_DATA_DIR / "drawings"))),
    "documents": Path(_env("DOCUMENTS_DATA_DIR", str(COMPANY_DATA_DIR / "documents"))),
}

for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

# ChromaDB persistent paths
_CHROMA_BASE = Path(_env("CHROMA_BASE_DIR", str(BASE_DIR / "chroma")))

CHROMA_PATHS = {
    "financial": str(_CHROMA_BASE / "financial"),
    "drawings":  str(_CHROMA_BASE / "drawings"),
    "documents": str(_CHROMA_BASE / "documents"),
}

for path in CHROMA_PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)

# Memoria persistente (JSON, semantic cards, watcher registry, financial DB...)
MEMORY_PATH = Path(_env("MEMORY_PATH", str(BASE_DIR / "memory")))
MEMORY_PATH.mkdir(parents=True, exist_ok=True)

# Cache markdown convertiti (PDF/DOCX/PPTX → MD)
MARKDOWN_CACHE_DIR = Path(_env("MARKDOWN_CACHE_DIR", str(BASE_DIR / "markdown_cache")))
MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Logs
LOGS_DIR = Path(_env("LOGS_DIR", str(BASE_DIR / "logs")))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM endpoint (OpenAI-compatible) ──────────────────────────────────────────
# Default: Ollama in dev (http://localhost:11434/v1).
# Produzione vLLM/SGLang: http://localhost:8000/v1 (o quello che si configura).
# L'API key è ignorata da Ollama ma OpenAI client la richiede non vuota.

LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY  = _env("LLM_API_KEY",  "ollama-no-key")

# Modello principale: usato per le risposte agli utenti.
# Dev: qwen2.5:7b. Prod (RTX 4090): qwen2.5:32b oppure qwen3:30b-a3b (MoE).
LLM_MODEL_MAIN = _env("LLM_MODEL_MAIN", "qwen2.5:7b")

# Modello fast: usato per routing (selezione documenti, generazione SQL,
# generazione semantic cards). Deve essere leggero e veloce.
LLM_MODEL_FAST = _env("LLM_MODEL_FAST", "qwen3:0.6b")

# Embedding model (usato da ChromaDB se non si usa il default interno).
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text")

# Timeout secondi per la generazione completa di una risposta (non-streaming).
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 180)

# Timeout secondi per il routing model (più stretto: deve essere veloce).
LLM_ROUTING_TIMEOUT_SECONDS = _env_int("LLM_ROUTING_TIMEOUT_SECONDS", 30)

# ── OCR Vision Model (multimodale, locale via Ollama) ─────────────────────────
# Modello vision dedicato all'estrazione di testo/tabelle/formule da immagini
# e PDF scansionati. Default: GLM-OCR (zai-org), ~0.9B parametri.
# Air-gapped: scaricare una volta con `ollama pull glm-ocr`, poi gira solo
# in locale come gli altri modelli Ollama. Stesso endpoint LLM_BASE_URL.
#
# Se OCR_ENABLED=False, l'agente documents cade automaticamente sul vecchio
# pipeline pytesseract → easyocr senza chiamare il modello vision.

OCR_ENABLED = _env_bool("OCR_ENABLED", True)
OCR_MODEL   = _env("OCR_MODEL", "glm-ocr")

# Soglia minima caratteri per accettare l'output OCR di un livello e non
# passare al successivo (corrisponde al valore già usato in documents_agent).
OCR_MIN_TEXT_LEN = _env_int("OCR_MIN_TEXT_LEN", 15)

# DPI di rasterizzazione delle pagine PDF scansionate prima di mandarle al
# modello vision. 200 DPI è un buon compromesso qualità/velocità; alzare a
# 300 per documenti con testo molto piccolo.
OCR_PDF_PAGE_RASTER_DPI = _env_int("OCR_PDF_PAGE_RASTER_DPI", 200)

# Sotto questa soglia di caratteri estratti da fitz, una pagina PDF viene
# considerata "scansionata" e rasterizzata + mandata a GLM-OCR.
OCR_PDF_MIN_TEXT_PER_PAGE = _env_int("OCR_PDF_MIN_TEXT_PER_PAGE", 50)

# Timeout (secondi) per una singola chiamata OCR vision. GLM-OCR è veloce
# (~1-2 sec/pagina su 4090), quindi 60s è ampiamente sufficiente.
OCR_TIMEOUT_SECONDS = _env_int("OCR_TIMEOUT_SECONDS", 60)

# ── Server ────────────────────────────────────────────────────────────────────

AGENT_PORT = _env_int("AGENT_PORT", 8000)
KIWIX_PORT = _env_int("KIWIX_PORT", 8080)

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE    = _env_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 60)

# ── File extensions per agente ────────────────────────────────────────────────

EXTENSIONS = {
    "financial": [".xlsx", ".xls", ".csv", ".pdf", ".txt"],
    "drawings":  [".dxf", ".dwg", ".svg", ".ifc",
                  ".stp", ".step", ".stl", ".obj", ".3dm", ".pdf"],
    "documents": [".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt", ".md"],
}

# ── Compatibilità retro (deprecati ma temporaneamente esposti) ────────────────
# Qualunque vecchio import che ancora usa OLLAMA_URL o LLM_MODEL continuerà a
# funzionare. Da rimuovere quando tutti i moduli saranno migrati.

OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")  # legacy
LLM_MODEL  = LLM_MODEL_MAIN                                 # legacy alias

# ── Multi-step retrieval (MSA-inspired) ──
MULTI_STEP_ENABLED         = os.getenv("MULTI_STEP_ENABLED", "true").lower() == "true"
MULTI_STEP_MAX_ROUNDS      = int(os.getenv("MULTI_STEP_MAX_ROUNDS", "1"))
MULTI_STEP_MIN_CONTEXT_LEN = int(os.getenv("MULTI_STEP_MIN_CONTEXT_LEN", "100"))
# ── Multi-step retrieval (MSA-inspired) ──
MULTI_STEP_ENABLED         = os.getenv("MULTI_STEP_ENABLED", "true").lower() == "true"
MULTI_STEP_MAX_ROUNDS      = int(os.getenv("MULTI_STEP_MAX_ROUNDS", "1"))
MULTI_STEP_MIN_CONTEXT_LEN = int(os.getenv("MULTI_STEP_MIN_CONTEXT_LEN", "100"))
