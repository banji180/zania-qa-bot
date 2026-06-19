"""Load and validate the list of questions from various input shapes.

Accepts either a JSON array of strings (``["q1", "q2"]``) or a JSON object with
a top-level ``questions`` key (``{"questions": ["q1", "q2"]}``). Both forms show
up in uploaded `.json` question files, so we normalise them here.
"""

from __future__ import annotations

import json


class QuestionLoadError(ValueError):
    """Raised when questions cannot be parsed or fail validation."""


def parse_questions(raw: str | bytes, *, max_questions: int) -> list[str]:
    """Parse questions from a raw JSON payload.

    Args:
        raw: JSON text — either an array of strings or an object with a
            ``questions`` array.
        max_questions: Upper bound on the number of questions allowed.

    Returns:
        A cleaned, de-duplicated (order-preserving) list of question strings.

    Raises:
        QuestionLoadError: If the payload is not valid JSON or has the wrong
            shape, is empty, or exceeds ``max_questions``.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QuestionLoadError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("questions")

    if not isinstance(data, list):
        raise QuestionLoadError(
            "Expected a JSON array of questions or an object with a "
            "'questions' array."
        )

    return normalize_questions(data, max_questions=max_questions)


def normalize_questions(items: list, *, max_questions: int) -> list[str]:
    """Validate and clean an already-parsed list of questions."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise QuestionLoadError(f"Question is not a string: {item!r}")
        q = item.strip()
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        cleaned.append(q)

    if not cleaned:
        raise QuestionLoadError("No questions provided.")
    if len(cleaned) > max_questions:
        raise QuestionLoadError(
            f"Too many questions: {len(cleaned)} (max {max_questions})."
        )
    return cleaned
