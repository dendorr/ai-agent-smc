"""Multi-agent OpenAI-compatible API server.

This server exposes three local agents through OpenAI-compatible endpoints:

- agent-drawings: technical drawings and CAD-related files
- agent-financial: spreadsheets, CSV files, and financial documents
- agent-documents: general business documents

The server is designed to be used behind Open WebUI and a local LLM backend
such as Ollama, vLLM, or SGLang.
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import ipaddress
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator


# Make imports work both when running from the repository root and from scripts/.
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

for path in (REPO_ROOT, CURRENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config.config import (  # noqa: E402
    AGENT_API_KEY,
    AGENT_HOST,
    AGENT_PORT,
    CORS_ORIGINS,
)


# Logging ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("server")


# Safety and performance limits ------------------------------------------------

MAX_MSG_CHARS = 8_000
MAX_HISTORY_MSGS = 20
MAX_HISTORY_CHARS = 10_000


# Agent metadata ---------------------------------------------------------------

AGENT_DESCRIPTIONS = {
    "agent-drawings": (
        "Technical drawings: DXF, STP, IFC, SVG, STL, technical PDFs, "
        "geometry, layers, and materials"
    ),
    "agent-financial": (
        "Financial documents: Excel, CSV, financial PDFs, budgets, invoices, "
        "and tabular data"
    ),
    "agent-documents": (
        "Business documents: PDF, PPTX, Word, Markdown, reports, "
        "presentations, and manuals"
    ),
}

AGENT_MODULES = {
    "agent-drawings": "drawings_agent",
    "agent-financial": "financial_agent",
    "agent-documents": "documents_agent",
}

AGENTS: dict[str, Any] = {}


def load_agents() -> None:
    """Load agent modules without preventing the server from starting.

    If one agent fails during import, the remaining agents can still work.
    The failed agent will be reported as unavailable in /health and /v1/models.
    """
    for agent_id, module_name in AGENT_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            count = module.collection.count()
            AGENTS[agent_id] = module
            logger.info("Loaded %s with %s indexed chunks", agent_id, f"{count:,}")
        except Exception as exc:
            logger.error("Failed to load %s: %s", agent_id, exc, exc_info=True)


load_agents()


# Authentication ---------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Validate Bearer token when AGENT_API_KEY is configured.

    When AGENT_API_KEY is empty the check is skipped entirely so existing
    local setups without a key continue to work without any change.

    The comparison is constant-time and done on bytes, so non-ASCII input
    is rejected with 401 instead of raising. The key is never logged.
    """
    if not AGENT_API_KEY:
        return

    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"),
        AGENT_API_KEY.encode("utf-8"),
    ):
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(
            "Rejected unauthenticated request from %s to %s",
            client_ip,
            request.url.path,
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _is_loopback_host(host: str) -> bool:
    """Return True when host binds only to the loopback interface."""
    if host == "localhost":
        return True

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# Application lifecycle --------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Close the shared LLM client during shutdown."""
    yield

    try:
        from llm_client import close_client

        await close_client()
        logger.info("LLM client closed.")
    except Exception as exc:
        logger.warning("Error while closing LLM client: %s", exc)


# FastAPI app ------------------------------------------------------------------

app = FastAPI(
    title="AI Agent SMC Server",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

# No origins configured means no CORS headers at all: browsers on other
# origins cannot read responses.
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )


# Request models ---------------------------------------------------------------


class Message(BaseModel):
    """Single OpenAI-compatible chat message."""

    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        """Accept only supported chat roles."""
        if value not in {"user", "assistant", "system"}:
            raise ValueError(
                f"Invalid role: {value!r}. Use user, assistant, or system."
            )

        return value

    @field_validator("content")
    @classmethod
    def truncate_content(cls, value: str) -> str:
        """Truncate overlong messages to protect local resources."""
        if len(value) > MAX_MSG_CHARS:
            logger.warning(
                "Message truncated from %s to %s characters",
                f"{len(value):,}",
                f"{MAX_MSG_CHARS:,}",
            )
            return value[:MAX_MSG_CHARS]

        return value


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


# Conversation helpers ---------------------------------------------------------


def extract_question_and_history(messages: list[Message]) -> tuple[str, str]:
    """Extract the latest user question and a truncated conversation history."""
    question = ""

    for message in reversed(messages):
        if message.role == "user":
            question = message.content
            break

    if not question:
        return "", ""

    if messages and messages[-1].role == "user":
        prior_messages = messages[:-1]
    else:
        prior_messages = messages[:]

    recent_messages = prior_messages[-MAX_HISTORY_MSGS:]
    history_parts: list[str] = []
    total_chars = 0

    for message in recent_messages:
        label = "User" if message.role == "user" else "Assistant"
        line = f"[{label}]: {message.content}"

        if total_chars + len(line) > MAX_HISTORY_CHARS:
            break

        history_parts.append(line)
        total_chars += len(line)

    return question, "\n".join(history_parts)


def build_full_question(question: str, history: str) -> str:
    """Combine conversation history and current question for the agent."""
    if not history:
        return question

    return (
        "=== CONVERSATION HISTORY ===\n"
        f"{history}\n\n"
        "=== CURRENT QUESTION ===\n"
        f"{question}"
    )


# SSE helpers ------------------------------------------------------------------


def sse_chunk(content: str, model: str, request_id: str, finish: bool = False) -> str:
    """Format a single OpenAI-compatible SSE chat completion chunk."""
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "delta": {"content": content} if not finish else {},
                "finish_reason": "stop" if finish else None,
                "index": 0,
            }
        ],
    }

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# Endpoints --------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return server health and per-agent status.

    The endpoint is unauthenticated, so it exposes only ok/error per agent.
    Error details are logged server-side.
    """
    agent_stats: dict[str, str] = {}

    for name in AGENT_DESCRIPTIONS:
        agent = AGENTS.get(name)

        if agent is None:
            logger.warning("Health check: agent %s is not loaded", name)
            agent_stats[name] = "error"
            continue

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, agent.collection.count)
            agent_stats[name] = "ok"
        except Exception as exc:
            logger.error("Health check failed for %s: %s", name, exc, exc_info=True)
            agent_stats[name] = "error"

    all_ok = all(status == "ok" for status in agent_stats.values())

    return {
        "status": "ok" if all_ok else "degraded",
        "agents": agent_stats,
    }


@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models() -> dict[str, Any]:
    """List available agents in an OpenAI-compatible model format."""
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "description": AGENT_DESCRIPTIONS.get(name, ""),
                "status": "ready" if name in AGENTS else "unavailable",
            }
            for name in AGENT_DESCRIPTIONS
        ],
    }


@app.post(
    "/v1/chat/completions",
    response_model=None,
    dependencies=[Depends(require_auth)],
)
async def chat(req: ChatRequest, request: Request) -> dict[str, Any] | StreamingResponse:
    """Handle OpenAI-compatible chat completions.

    The selected agent is determined by req.model.

    Streaming requests return Server-Sent Events.
    Non-streaming requests return a complete OpenAI-compatible response object.
    """
    del request

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    start_time = time.perf_counter()

    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    agent = AGENTS.get(req.model)

    if not agent:
        known_agents = list(AGENT_DESCRIPTIONS.keys())
        available_agents = list(AGENTS.keys())

        detail = f"Agent {req.model!r} was not found."

        if req.model in known_agents and req.model not in AGENTS:
            detail += (
                " The agent is known but failed to load during startup. "
                "Check server logs for details."
            )

        detail += f" Available agents: {available_agents}"

        raise HTTPException(status_code=404, detail=detail)

    question, history = extract_question_and_history(req.messages)

    if not question:
        raise HTTPException(status_code=400, detail="No user message found.")

    full_question = build_full_question(question, history)

    if req.stream:

        async def event_generator():
            try:
                # Search uses the raw user question for better retrieval quality.
                context = await agent.search(question)

                logger.info(
                    "[%s] [%s] stream | history=%s chars | question=%s...",
                    request_id[:12],
                    req.model,
                    len(history),
                    question[:60],
                )

                async for token in agent.answer_stream(full_question, context):
                    yield sse_chunk(token, req.model, request_id)

                yield sse_chunk("", req.model, request_id, finish=True)
                yield "data: [DONE]\n\n"

                elapsed = time.perf_counter() - start_time
                logger.info(
                    "[%s] [%s] completed in %.1fs",
                    request_id[:12],
                    req.model,
                    elapsed,
                )

            except Exception as exc:
                logger.error(
                    "[%s] [%s] streaming error: %s",
                    request_id[:12],
                    req.model,
                    exc,
                    exc_info=True,
                )

                error_payload = json.dumps(
                    {"error": {"message": str(exc), "type": "agent_error"}},
                    ensure_ascii=False,
                )
                yield f"data: {error_payload}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "X-Request-Id": request_id,
            },
        )

    try:
        context = await agent.search(question)
        response = await agent.answer(full_question, context)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "[%s] [%s] %.1fs | history=%s chars | question=%s...",
            request_id[:12],
            req.model,
            elapsed,
            len(history),
            question[:60],
        )

        prompt_tokens = len(full_question.split())
        completion_tokens = len(response.split())

        return {
            "id": request_id,
            "object": "chat.completion",
            "model": req.model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    except Exception as exc:
        logger.error(
            "[%s] [%s] error: %s",
            request_id[:12],
            req.model,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Agent {req.model!r} error: {exc}",
        ) from exc


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON error response for unexpected exceptions."""
    logger.error(
        "Unhandled exception on %s: %s",
        request.url,
        exc,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("AI Agent SMC server starting")
    logger.info("Loaded agents: %s", list(AGENTS.keys()))

    if len(AGENTS) < len(AGENT_DESCRIPTIONS):
        missing_agents = [name for name in AGENT_DESCRIPTIONS if name not in AGENTS]
        logger.warning("Missing agents: %s", missing_agents)

    logger.info("Host: %s  Port: %s", AGENT_HOST, AGENT_PORT)
    logger.info("API key auth: %s", "enabled" if AGENT_API_KEY else "disabled")
    logger.info("=" * 60)

    if not AGENT_API_KEY and not _is_loopback_host(AGENT_HOST):
        logger.error(
            "Refusing to start: AGENT_HOST=%s is not loopback and "
            "AGENT_API_KEY is empty. Set AGENT_API_KEY or use 127.0.0.1.",
            AGENT_HOST,
        )
        sys.exit(1)

    if AGENT_API_KEY and len(AGENT_API_KEY) < 32:
        logger.warning(
            "AGENT_API_KEY is shorter than 32 characters. Generate one with: "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    uvicorn.run(
        app,
        host=AGENT_HOST,
        port=AGENT_PORT,
        access_log=True,
        loop="asyncio",
    )
