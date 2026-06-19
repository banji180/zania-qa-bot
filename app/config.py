"""Application configuration.

Settings are read from the environment (and an optional `.env` file). See
`.env.example` for the full list of supported variables.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    # Using gpt-4o-mini for cost efficiency
    openai_model: str = "gpt-4o-mini"

    # RAG / Retrieval
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3  # Number of chunks to retrieve per question

    # Behaviour
    # Text returned when an answer cannot be grounded in the document.
    not_found_text: str = "Data Not Available"

    # Limits
    max_questions: int = 50
    max_document_bytes: int = 32 * 1024 * 1024  # 32 MB


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
