"""Shared OpenAI-compatible LLM clients and helper functions.

This module centralizes all model calls for the agents. It supports any
OpenAI-compatible backend configured through the application settings, such as
Ollama, vLLM, or SGLang.

Public helpers:
- chat_complete: asynchronous non-streaming chat completion.
- chat_complete_stream: asynchronous streaming chat completion.
- chat_complete_json: asynchronous chat completion tuned for JSON outputs.
- vision_extract: asynchronous OCR/vision extraction.
- vision_extract_sync: synchronous OCR/vision extraction for indexing flows.
- close_client: graceful cleanup for shared clients.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from typing import Any, AsyncIterator, Optional

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (  # noqa: E402
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ROUTING_TIMEOUT_SECONDS,
    LLM_TIMEOUT_SECONDS,
    OCR_TIMEOUT_SECONDS,
)

logger = logging.getLogger("llm_client")

_client: Optional[AsyncOpenAI] = None
_sync_client: Optional[OpenAI] = None


def get_client() -> AsyncOpenAI:
    """Return the shared asynchronous OpenAI-compatible client."""
    global _client

    if _client is None:
        _client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info("Initialized async LLM client for %s", LLM_BASE_URL)

    return _client


def get_sync_client() -> OpenAI:
    """Return the shared synchronous OpenAI-compatible client."""
    global _sync_client

    if _sync_client is None:
        _sync_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info("Initialized sync LLM client for %s", LLM_BASE_URL)

    return _sync_client


def _build_chat_messages(system: str, user: str) -> list[dict[str, str]]:
    """Build a standard system/user chat message list."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_vision_messages(
    image_bytes: bytes,
    prompt: str,
    image_format: str,
) -> list[dict[str, Any]]:
    """Build an OpenAI-compatible vision message payload with a data URL."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/{image_format};base64,{image_b64}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


async def chat_complete(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> str:
    """Generate a complete non-streaming chat response.

    Errors are caught and returned as readable strings so callers can decide
    whether to display them to the user.
    """
    client = get_client()
    request_timeout = timeout if timeout is not None else LLM_TIMEOUT_SECONDS

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=_build_chat_messages(system=system, user=user),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
        return response.choices[0].message.content or ""

    except APITimeoutError:
        logger.warning("LLM request timed out for model %s", model)
        return "Timeout — prova una domanda più specifica."

    except APIConnectionError as exc:
        logger.error("LLM connection failed for model %s: %s", model, exc)
        return f"Errore di connessione al modello LLM: {exc}"

    except APIError as exc:
        logger.error("LLM API error for model %s: %s", model, exc)
        return f"Errore del modello: {exc}"

    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.error("Unexpected LLM error for model %s: %s", model, exc, exc_info=True)
        return f"Errore: {exc}"


async def chat_complete_stream(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """Generate a streaming chat response, yielding text chunks as they arrive."""
    client = get_client()

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=_build_chat_messages(system=system, user=user),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
            except (IndexError, AttributeError):
                continue

            if content:
                yield content

    except APITimeoutError:
        logger.warning("Streaming LLM request timed out for model %s", model)
        yield "\n\n[Timeout del modello — prova una domanda più specifica.]"

    except APIConnectionError as exc:
        logger.error("Streaming LLM connection failed for model %s: %s", model, exc)
        yield f"\n\n[Errore di connessione al modello: {exc}]"

    except APIError as exc:
        logger.error("Streaming LLM API error for model %s: %s", model, exc)
        yield f"\n\n[Errore del modello: {exc}]"

    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.error(
            "Unexpected streaming LLM error for model %s: %s",
            model,
            exc,
            exc_info=True,
        )
        yield f"\n\n[Errore: {exc}]"


async def chat_complete_json(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """Generate a chat response intended to contain JSON.

    The request does not force OpenAI's JSON response format because some
    OpenAI-compatible backends do not support it consistently.
    """
    request_timeout = timeout if timeout is not None else LLM_ROUTING_TIMEOUT_SECONDS

    return await chat_complete(
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        timeout=request_timeout,
    )


async def vision_extract(
    model: str,
    image_bytes: bytes,
    prompt: str = "Text Recognition:",
    image_format: str = "png",
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """Extract text from an image using an OpenAI-compatible vision model.

    Returns an empty string on failure so callers can fallback to another OCR
    implementation without handling exceptions.
    """
    if not image_bytes:
        return ""

    client = get_client()
    request_timeout = timeout if timeout is not None else OCR_TIMEOUT_SECONDS

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=_build_vision_messages(
                image_bytes=image_bytes,
                prompt=prompt,
                image_format=image_format,
            ),
            temperature=temperature,
            timeout=request_timeout,
        )
        return (response.choices[0].message.content or "").strip()

    except APITimeoutError:
        logger.warning("Async vision OCR timed out for model %s", model)
        return ""

    except APIConnectionError as exc:
        logger.error("Async vision OCR connection failed for model %s: %s", model, exc)
        return ""

    except APIError as exc:
        logger.error("Async vision OCR API error for model %s: %s", model, exc)
        return ""

    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.error(
            "Unexpected async vision OCR error for model %s: %s",
            model,
            exc,
            exc_info=True,
        )
        return ""


def vision_extract_sync(
    model: str,
    image_bytes: bytes,
    prompt: str = "Text Recognition:",
    image_format: str = "png",
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """Synchronous version of vision_extract for indexing pipelines."""
    if not image_bytes:
        return ""

    client = get_sync_client()
    request_timeout = timeout if timeout is not None else OCR_TIMEOUT_SECONDS

    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_vision_messages(
                image_bytes=image_bytes,
                prompt=prompt,
                image_format=image_format,
            ),
            temperature=temperature,
            timeout=request_timeout,
        )
        return (response.choices[0].message.content or "").strip()

    except APITimeoutError:
        logger.warning("Sync vision OCR timed out for model %s", model)
        return ""

    except APIConnectionError as exc:
        logger.error("Sync vision OCR connection failed for model %s: %s", model, exc)
        return ""

    except APIError as exc:
        logger.error("Sync vision OCR API error for model %s: %s", model, exc)
        return ""

    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.error(
            "Unexpected sync vision OCR error for model %s: %s",
            model,
            exc,
            exc_info=True,
        )
        return ""


async def close_client() -> None:
    """Close shared clients and reset their singleton references."""
    global _client, _sync_client

    if _client is not None:
        try:
            await _client.close()
        except Exception as exc:  # pragma: no cover - cleanup best effort
            logger.debug("Failed to close async LLM client cleanly: %s", exc)
        finally:
            _client = None

    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception as exc:  # pragma: no cover - cleanup best effort
            logger.debug("Failed to close sync LLM client cleanly: %s", exc)
        finally:
            _sync_client = None
