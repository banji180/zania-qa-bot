"""Pydantic models for requests, responses, and LLM output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """A single answer grounded in the source document."""

    question: str = Field(description="The question, echoed verbatim.")
    answer: str = Field(
        description=(
            "The answer drawn from the document, or the configured "
            "not-found text when the document does not contain it."
        )
    )
    found: bool = Field(
        description="True when the answer was located in the document."
    )


class AskRequest(BaseModel):
    """Request body for the JSON `/ask` endpoint (document supplied as text)."""

    document: str = Field(description="Plain-text document to answer over.")
    questions: list[str] = Field(min_length=1)


class AskResponse(BaseModel):
    """Response body returned by the `/ask` endpoints."""

    answers: list[Answer]

