"""
LLM CLIENT — singleton AsyncOpenAI (+ OpenAI sync) condiviso da tutti gli agenti

Punto unico di configurazione per le chiamate al modello.
Funziona out-of-the-box con qualsiasi backend OpenAI-compatibile:

  Dev    : Ollama         → LLM_BASE_URL=http://localhost:11434/v1
  Prod A : vLLM           → LLM_BASE_URL=http://localhost:8000/v1
  Prod B : SGLang         → LLM_BASE_URL=http://localhost:30000/v1

Espone funzioni helper:
  - chat_complete()        : risposta intera (non-streaming, async)
  - chat_complete_stream() : streaming async dei token
  - chat_complete_json()   : risposta forzata in JSON (per routing/SQL)
  - vision_extract()       : OCR vision async (per contesti async)
  - vision_extract_sync()  : OCR vision sync (per pipeline di indexing sync)

Uso async (da un agente):

    from llm_client import chat_complete
    text = await chat_complete(
        model=ANSWER_MODEL,
        system="Sei un assistente...",
        user="Domanda dell'utente",
    )

Uso sync (da watcher/indexing):

    from llm_client import vision_extract_sync
    text = vision_extract_sync(model="glm-ocr", image_bytes=img_bytes)
"""

import sys
import os
import logging
import base64
from typing import AsyncIterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TIMEOUT_SECONDS,
    LLM_ROUTING_TIMEOUT_SECONDS,
    OCR_TIMEOUT_SECONDS,
)

from openai import AsyncOpenAI, OpenAI
from openai import APIConnectionError, APITimeoutError, APIError

logger = logging.getLogger("llm_client")

# ── Singleton clients ─────────────────────────────────────────────────────────
# Due istanze condivise (async + sync). Il connection pool sottostante riusa
# le connessioni HTTP, riducendo latenza e overhead per richieste multiple.
# Il client sync è usato dal pipeline di indexing (watcher → index_file → OCR)
# che gira fuori dall'event loop e non può fare await.

_client: Optional[AsyncOpenAI] = None
_sync_client: Optional[OpenAI] = None


def get_client() -> AsyncOpenAI:
    """Restituisce il client AsyncOpenAI condiviso (lazy-init)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info(f"LLM async client inizializzato → {LLM_BASE_URL}")
    return _client


def get_sync_client() -> OpenAI:
    """
    Restituisce il client OpenAI sync condiviso (lazy-init).
    Usato dal pipeline di indexing per chiamate vision/OCR fuori dall'event loop.
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info(f"LLM sync client inizializzato → {LLM_BASE_URL}")
    return _sync_client


# ── Chat completion (non-streaming) ───────────────────────────────────────────

async def chat_complete(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> str:
    """
    Genera una risposta completa (non-streaming).

    Restituisce il testo della risposta o un messaggio di errore leggibile.
    Non solleva eccezioni: gli errori vengono catturati e ritornati come stringa
    in modo che il chiamante (agent) possa decidere se mostrarli o meno.
    """
    client = get_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout if timeout is not None else LLM_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content or ""

    except APITimeoutError:
        logger.warning(f"Timeout LLM ({model})")
        return "Timeout — prova una domanda più specifica."
    except APIConnectionError as e:
        logger.error(f"Connessione LLM fallita ({model}): {e}")
        return f"Errore di connessione al modello LLM: {e}"
    except APIError as e:
        logger.error(f"Errore API LLM ({model}): {e}")
        return f"Errore del modello: {e}"
    except Exception as e:
        logger.error(f"Errore imprevisto LLM ({model}): {e}", exc_info=True)
        return f"Errore: {e}"


# ── Chat completion streaming ─────────────────────────────────────────────────

async def chat_complete_stream(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """
    Genera una risposta in streaming, yielding chunk di testo non appena
    arrivano dal modello. Usato dall'endpoint /v1/chat/completions con
    stream=True per inoltrare i token a Open WebUI in tempo reale.

    In caso di errore, yield un singolo messaggio testuale e termina.
    """
    client = get_client()

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
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
                # Chunk senza delta valido (es: ruolo iniziale) → ignoriamo
                continue

    except APITimeoutError:
        logger.warning(f"Timeout streaming LLM ({model})")
        yield "\n\n[Timeout del modello — prova una domanda più specifica.]"
    except APIConnectionError as e:
        logger.error(f"Connessione streaming LLM fallita ({model}): {e}")
        yield f"\n\n[Errore di connessione al modello: {e}]"
    except APIError as e:
        logger.error(f"Errore API streaming LLM ({model}): {e}")
        yield f"\n\n[Errore del modello: {e}]"
    except Exception as e:
        logger.error(f"Errore imprevisto streaming LLM ({model}): {e}", exc_info=True)
        yield f"\n\n[Errore: {e}]"


# ── Chat completion JSON (per routing) ────────────────────────────────────────

async def chat_complete_json(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """
    Variante usata per la generazione di output JSON (routing model).

    Timeout più stretto rispetto a chat_complete (default = LLM_ROUTING_TIMEOUT_SECONDS).
    Temperature = 0 per output deterministico.

    NB: non forziamo response_format={"type": "json_object"} perché Ollama
        non lo supporta su tutti i modelli; ci affidiamo al prompt + a
        un parsing tollerante nel chiamante.
    """
    return await chat_complete(
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        timeout=timeout if timeout is not None else LLM_ROUTING_TIMEOUT_SECONDS,
    )


# ── Vision OCR (async + sync) ─────────────────────────────────────────────────
# Helpers per modelli vision multimodali (es: GLM-OCR via Ollama).
# Ritornano stringa vuota in caso di errore — il chiamante può fare fallback
# su altri OCR (pytesseract, easyocr) senza dover gestire eccezioni.


def _build_vision_messages(image_bytes: bytes, prompt: str, image_format: str) -> list:
    """Costruisce il payload OpenAI vision (data URL base64)."""
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
    """
    Estrae testo da un'immagine usando un modello vision (es. GLM-OCR).
    Versione async — usabile da contesti già asyncroni (server, search).

    Prompts supportati da GLM-OCR:
      - "Text Recognition:"     → riconoscimento testo (default)
      - "Formula Recognition:"  → riconoscimento formule matematiche
      - "Table Recognition:"    → riconoscimento tabelle (output Markdown)
      - schema JSON             → information extraction strutturata

    Ritorna stringa vuota in caso di errore (non solleva eccezioni).
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
        logger.warning(f"Timeout vision OCR async ({model})")
        return ""
    except APIConnectionError as e:
        logger.error(f"Connessione vision async fallita ({model}): {e}")
        return ""
    except APIError as e:
        logger.error(f"Errore API vision async ({model}): {e}")
        return ""
    except Exception as e:
        logger.error(f"Errore imprevisto vision async ({model}): {e}", exc_info=True)
        return ""


def vision_extract_sync(
    model: str,
    image_bytes: bytes,
    prompt: str = "Text Recognition:",
    image_format: str = "png",
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """
    Versione sync di vision_extract — usabile dal pipeline di indexing
    (watcher → index_file → OCR), che gira fuori dall'event loop e non
    può fare await.

    Stessa interfaccia e stesso comportamento di vision_extract,
    ma usa il client sync. Ritorna stringa vuota in caso di errore.
    """
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
        logger.warning(f"Timeout vision OCR sync ({model})")
        return ""
    except APIConnectionError as e:
        logger.error(f"Connessione vision sync fallita ({model}): {e}")
        return ""
    except APIError as e:
        logger.error(f"Errore API vision sync ({model}): {e}")
        return ""
    except Exception as e:
        logger.error(f"Errore imprevisto vision sync ({model}): {e}", exc_info=True)
        return ""


# ── Cleanup ───────────────────────────────────────────────────────────────────

async def close_client():
    """Chiude entrambi i client (chiamare allo shutdown del server)."""
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