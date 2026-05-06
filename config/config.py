"""Central configuration for AI Agent SMC.

All runtime settings are controlled through environment variables.
Defaults are designed for local development with Ollama and repository-local
runtime folders.

Production deployments should override paths and LLM settings through .env,
systemd, Docker, or shell exports.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Return an environment variable, falling back to default if empty."""
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable with a safe fallback."""
    value = os.environ.get(name, "").strip()

    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable with common truthy values."""
    value = os.environ.get(name, "").strip().lower()

    if not value:
        return default

    return value in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path | str) -> Path:
    """Return an expanded absolute Path from an environment variable."""
    raw_value = _env(name, str(default))
    return Path(raw_value).expanduser().resolve()


# Repository and runtime paths.
#
# REPO_ROOT is inferred from this file:
#   <repo>/config/config.py -> <repo>
#
# This avoids hardcoding ~/ai-agent or ~/ai-agent-smc.
HOME_DIR = Path(os.path.expanduser("~"))
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = _env_path("AI_AGENT_BASE_DIR", REPO_ROOT)

# Company data directory.
#
# Default:
#   <repo>/data
#
# Production deployments can point this to a mounted network folder or
# another local disk path through COMPANY_DATA_DIR.
COMPANY_DATA_DIR = _env_path("COMPANY_DATA_DIR", BASE_DIR / "data")

FOLDERS = {
    "financial": _env_path("FINANCIAL_DATA_DIR", COMPANY_DATA_DIR / "financial"),
    "drawings": _env_path("DRAWINGS_DATA_DIR", COMPANY_DATA_DIR / "drawings"),
    "documents": _env_path("DOCUMENTS_DATA_DIR", COMPANY_DATA_DIR / "documents"),
}

# ChromaDB persistent storage.
CHROMA_BASE_DIR = _env_path("CHROMA_BASE_DIR", BASE_DIR / "chroma")
_CHROMA_BASE = CHROMA_BASE_DIR  # Backward-compatible private alias.

CHROMA_PATHS = {
    "financial": str(CHROMA_BASE_DIR / "financial"),
    "drawings": str(CHROMA_BASE_DIR / "drawings"),
    "documents": str(CHROMA_BASE_DIR / "documents"),
}

# Persistent runtime folders.
MEMORY_PATH = _env_path("MEMORY_PATH", BASE_DIR / "memory")
MARKDOWN_CACHE_DIR = _env_path("MARKDOWN_CACHE_DIR", BASE_DIR / "markdown_cache")
LOGS_DIR = _env_path("LOGS_DIR", BASE_DIR / "logs")


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they do not exist."""
    for folder in FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    for chroma_path in CHROMA_PATHS.values():
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

    MEMORY_PATH.mkdir(parents=True, exist_ok=True)
    MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()


# LLM endpoint.
#
# Development default:
#   Ollama OpenAI-compatible endpoint
#   http://localhost:11434/v1
#
# Production examples:
#   vLLM   -> http://localhost:8001/v1
#   SGLang -> http://localhost:30000/v1
#
# Ollama ignores the API key, but the OpenAI client requires a non-empty value.
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _env("LLM_API_KEY", "ollama-no-key")

# Main model: used for final user-facing answers.
LLM_MODEL_MAIN = _env("LLM_MODEL_MAIN", "qwen2.5:7b")

# Fast model: used for routing, SQL generation, document selection,
# context evaluation, and semantic cards.
LLM_MODEL_FAST = _env("LLM_MODEL_FAST", "qwen3:0.6b")

# Embedding model name.
#
# Actual behavior depends on the ChromaDB embedding setup used by the agents.
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text")

# LLM timeouts.
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 180)
LLM_ROUTING_TIMEOUT_SECONDS = _env_int("LLM_ROUTING_TIMEOUT_SECONDS", 30)


# OCR settings.
#
# OCR_ENABLED controls whether the documents agent can use the vision OCR path.
# If disabled, the agent should fall back to local CPU OCR where implemented.
OCR_ENABLED = _env_bool("OCR_ENABLED", True)

# Default OCR model.
#
# GLM-OCR is intended for local vision OCR through an Ollama/vLLM/SGLang adapter.
OCR_MODEL = _env("OCR_MODEL", "glm-ocr")

# Minimum accepted text length for OCR output before falling back to another tier.
OCR_MIN_TEXT_LEN = _env_int("OCR_MIN_TEXT_LEN", 15)

# Rasterization DPI for scanned PDF pages before OCR.
OCR_PDF_PAGE_RASTER_DPI = _env_int("OCR_PDF_PAGE_RASTER_DPI", 200)

# If native PDF extraction returns less text than this threshold for a page,
# the page can be treated as scanned and sent to OCR.
OCR_PDF_MIN_TEXT_PER_PAGE = _env_int("OCR_PDF_MIN_TEXT_PER_PAGE", 50)

# Timeout for a single OCR model call.
OCR_TIMEOUT_SECONDS = _env_int("OCR_TIMEOUT_SECONDS", 60)


# Server settings.
AGENT_PORT = _env_int("AGENT_PORT", 8000)
KIWIX_PORT = _env_int("KIWIX_PORT", 8080)


# Chunking settings.
CHUNK_SIZE = _env_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 60)


# Supported file extensions by agent.
EXTENSIONS = {
    "financial": [
        ".xlsx",
        ".xls",
        ".csv",
        ".pdf",
        ".txt",
    ],
    "drawings": [
        ".dxf",
        ".dwg",
        ".svg",
        ".ifc",
        ".stp",
        ".step",
        ".stl",
        ".obj",
        ".3dm",
        ".pdf",
    ],
    "documents": [
        ".pdf",
        ".pptx",
        ".ppt",
        ".docx",
        ".doc",
        ".txt",
        ".md",
    ],
}


# Multi-step retrieval settings.
#
# This is inspired by iterative search / memory interleave patterns:
# first retrieve context, evaluate sufficiency with the fast model, then
# optionally issue one or more follow-up retrieval rounds.
MULTI_STEP_ENABLED = _env_bool("MULTI_STEP_ENABLED", True)
MULTI_STEP_MAX_ROUNDS = _env_int("MULTI_STEP_MAX_ROUNDS", 1)
MULTI_STEP_MIN_CONTEXT_LEN = _env_int("MULTI_STEP_MIN_CONTEXT_LEN", 100)


# Backward-compatible aliases.
#
# Keep these until all modules stop importing legacy names.
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = LLM_MODEL_MAIN