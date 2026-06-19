"""Turn raw document bytes into Anthropic message content blocks.

PDFs are passed natively to the model as base64 ``document`` blocks (Claude
reads PDFs directly, preserving layout and any embedded text). Plain-text
formats are passed as ``text`` blocks.
"""

from __future__ import annotations

import base64
from typing import Any

# Content type used for the cacheable document block. A single ephemeral
# breakpoint here lets repeated questions over the same document reuse the cache.
_CACHE_CONTROL = {"type": "ephemeral"}


def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def build_document_blocks(
    data: bytes,
    *,
    filename: str | None = None,
    media_type: str | None = None,
) -> list[dict[str, Any]]:
    """Build the document content block(s) for a single uploaded document.

    Args:
        data: Raw document bytes.
        filename: Original filename, used only as a hint for type detection.
        media_type: Explicit MIME type, if known.

    Returns:
        A list of content blocks ready to splice into a user message, with a
        cache breakpoint set on the final block.
    """
    if not data:
        raise ValueError("Document is empty.")

    is_pdf = (
        (media_type == "application/pdf")
        or (filename or "").lower().endswith(".pdf")
        or _looks_like_pdf(data)
    )

    if is_pdf:
        block: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
            "cache_control": _CACHE_CONTROL,
        }
        return [block]

    # Treat everything else as UTF-8 text.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Document is not a PDF and could not be decoded as UTF-8 text."
        ) from exc

    return [
        {
            "type": "text",
            "text": text,
            "cache_control": _CACHE_CONTROL,
        }
    ]


def build_text_document_blocks(text: str) -> list[dict[str, Any]]:
    """Build a document block from already-decoded text (JSON `/ask` path)."""
    if not text.strip():
        raise ValueError("Document text is empty.")
    return [{"type": "text", "text": text, "cache_control": _CACHE_CONTROL}]
