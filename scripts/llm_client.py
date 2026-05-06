"""Shared LLM client helpers.

This module centralizes OpenAI-compatible client configuration for all agents.
It supports local and production backends such as Ollama, vLLM, and SGLang.

Async helpers are used by agents and the FastAPI server. Sync helpers are used
by indexing pipelines that run outside the event loop.
"""

import base64
import logging
import os
import sys
from typing import AsyncIterator, Optional

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
    """Return the shared AsyncOpenAI client, creating it lazily."""
    global _client

    if _client is None:
        _client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info("Initialized async LLM client: %s", LLM_BASE_URL)

    return _client


def get_sync_client() -> OpenAI:
    """Return the shared sync OpenAI client, creating it lazily.

    The sync client is used by indexing pipelines such as watcher -> index_file
    -> OCR, where calls run outside the event loop.
    """
    global _sync_client

    if _sync_client is None:
        _sync_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info("Initialized sync LLM client: %s", LLM_BASE_URL)

    return _sync_client


async def chat_complete(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> str:
    """Generate a full non-streaming chat completion.

    Errors are caught and returned as readable strings so callers can decide
    whether to show them to the user.
    """
    client = get_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout if timeout is not None else LLM_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content or ""
    except APITimeoutError:
        logger.warning("LLM timeout (%s)", model)
        return "Timeout — prova una domanda più specifica."
    except APIConnectionError as exc:
        logger.error("LLM connection failed (%s): %s", model, exc)
        return f"Errore di connessione al modello LLM: {exc}"
    except APIError as exc:
        logger.error("LLM API error (%s): %s", model, exc)
        return f"Errore del modello: {exc}"
    except Exception as exc:
        logger.error("Unexpected LLM error (%s): %s", model, exc, exc_info=True)
        return f"Errore: {exc}"


async def chat_complete_stream(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """Generate a streaming chat completion.

    This is used by the /v1/chat/completions endpoint when stream=True. On
    error, it yields one readable message and then stops.
    """
    client = get_client()

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except (IndexError, AttributeError):
                continue
    except APITimeoutError:
        logger.warning("Streaming LLM timeout (%s)", model)
        yield "\n\n[Timeout del modello — prova una domanda più specifica.]"
    except APIConnectionError as exc:
        logger.error("Streaming LLM connection failed (%s): %s", model, exc)
        yield f"\n\n[Errore di connessione al modello: {exc}]"
    except APIError as exc:
        logger.error("Streaming LLM API error (%s): %s", model, exc)
        yield f"\n\n[Errore del modello: {exc}]"
    except Exception as exc:
        logger.error(
            "Unexpected streaming LLM error (%s): %s",
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
    """Generate JSON-oriented chat output for routing tasks.

    The default timeout is shorter than the standard chat helper. The function
    does not force response_format={"type": "json_object"} because some Ollama
    models do not support it consistently; callers should still parse output
    defensively.
    """
    return await chat_complete(
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        timeout=timeout if timeout is not None else LLM_ROUTING_TIMEOUT_SECONDS,
    )


def _build_vision_messages(image_bytes: bytes, prompt: str, image_format: str) -> list:
    """Build an OpenAI-compatible vision payload using a base64 data URL."""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/{image_format};base64,{img_b64}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


async def vision_extract(
    model: str,
    image_bytes: bytes,
    prompt: str = "Text Recognition:",
    image_format: str = "png",
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """Extract text from an image using a vision model.

    Returns an empty string on error so callers can fall back to another OCR
    path without handling exceptions.
    """
    if not image_bytes:
        return ""

    client = get_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=_build_vision_messages(image_bytes, prompt, image_format),
            temperature=temperature,
            timeout=timeout if timeout is not None else OCR_TIMEOUT_SECONDS,
        )
        return (response.choices[0].message.content or "").strip()
    except APITimeoutError:
        logger.warning("Async vision OCR timeout (%s)", model)
        return ""
    except APIConnectionError as exc:
        logger.error("Async vision connection failed (%s): %s", model, exc)
        return ""
    except APIError as exc:
        logger.error("Async vision API error (%s): %s", model, exc)
        return ""
    except Exception as exc:
        logger.error("Unexpected async vision error (%s): %s", model, exc, exc_info=True)
        return ""


def vision_extract_sync(
    model: str,
    image_bytes: bytes,
    prompt: str = "Text Recognition:",
    image_format: str = "png",
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """Sync version of vision_extract for indexing pipelines."""
    if not image_bytes:
        return ""

    client = get_sync_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_vision_messages(image_bytes, prompt, image_format),
            temperature=temperature,
            timeout=timeout if timeout is not None else OCR_TIMEOUT_SECONDS,
        )
        return (response.choices[0].message.content or "").strip()
    except APITimeoutError:
        logger.warning("Sync vision OCR timeout (%s)", model)
        return ""
    except APIConnectionError as exc:
        logger.error("Sync vision connection failed (%s): %s", model, exc)
        return ""
    except APIError as exc:
        logger.error("Sync vision API error (%s): %s", model, exc)
        return ""
    except Exception as exc:
        logger.error("Unexpected sync vision error (%s): %s", model, exc, exc_info=True)
        return ""


async def close_client() -> None:
    """Close shared clients during server shutdown."""
    global _client, _sync_client

    if _client is not None:
        try:
            await _client.close()
        except Exception:
            pass
        _client = None

    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception:
            pass
        _sync_client = None
