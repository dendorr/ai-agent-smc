"""OCR helpers for the documents agent.

This module keeps OCR-related dependencies and fallback logic isolated from
``documents_agent.py``. All OCR runs locally and returns empty strings instead
of raising so callers can decide whether to continue with other extraction
methods.
"""

from pathlib import Path

from config.config import OCR_ENABLED, OCR_MIN_TEXT_LEN, OCR_MODEL
from llm_client import vision_extract_sync

# ── OCR — singleton (no re-init) ──────────────────────────────────────────────
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy-init easyocr Reader — once per process."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(["it", "en"], gpu=False, verbose=False)
        except Exception:
            pass
    return _easyocr_reader


# ── GLM-OCR vision (Tier 0) ──────────────────────────────────────────────────

def ocr_with_glm(image_bytes: bytes, source_hint: str = "") -> str:
    """
    OCR via GLM-OCR multimodal (Ollama, local).

    Expects image_bytes in PNG format (canonical). Returns the cleaned
    extracted text or empty string on error / OCR disabled.
    Does not raise exceptions: caller can fallback to other tiers.
    """
    if not image_bytes or not OCR_ENABLED:
        return ""
    try:
        text = vision_extract_sync(
            model=OCR_MODEL,
            image_bytes=image_bytes,
            prompt="Text Recognition:",
            image_format="png",
        )
        return text.strip() if text else ""
    except Exception:
        return ""


def ocr_image_bytes(image_bytes: bytes, source_hint: str = "") -> str:
    """
    OCR on raw bytes, with 3-tier cascade and automatic fallback.

      Tier 0: GLM-OCR (multimodal vision via Ollama) — top accuracy
              on complex layouts, tables, formulas. Skipped if OCR_ENABLED=False.
      Tier 1: pytesseract — fast, CPU, accurate for printed/scanned text
      Tier 2: easyocr     — neural, better for stylized/handwritten text

    Returns the extracted text block, or source_hint if nothing is readable.
    """
    if not image_bytes:
        return ""

    from PIL import Image
    import io

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception:
        return ""

    # Canonical conversion to PNG bytes — consistent format for all tiers
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
    except Exception:
        png_bytes = image_bytes  # fallback to original bytes

    # Tier 0: GLM-OCR
    if OCR_ENABLED:
        try:
            text = ocr_with_glm(png_bytes, source_hint=source_hint)
            if text and len(text) > OCR_MIN_TEXT_LEN:
                return f"[OCR]\n{text}"
        except Exception:
            pass

    # Tier 1: pytesseract
    try:
        import pytesseract
        text = pytesseract.image_to_string(
            img, lang="ita+eng", config="--psm 3 --oem 3"
        ).strip()
        if len(text) > OCR_MIN_TEXT_LEN:
            return f"[OCR]\n{text}"
    except Exception:
        pass

    # Tier 2: easyocr
    try:
        import numpy as np
        reader = _get_easyocr_reader()
        if reader:
            results = reader.readtext(np.array(img))
            lines = [r[1] for r in results if len(r) > 2 and r[2] > 0.3]
            text = "\n".join(lines).strip()
            if len(text) > OCR_MIN_TEXT_LEN:
                return f"[OCR]\n{text}"
    except Exception:
        pass

    return source_hint


def ocr_image_file(filepath) -> str:
    """OCR on a file path."""
    try:
        return ocr_image_bytes(Path(filepath).read_bytes())
    except Exception:
        return ""
