"""The question-answering service using Llama Index for RAG.

Uses vector retrieval to find relevant chunks of the document, then sends
only those chunks to the LLM. This reduces token usage and cost compared to
sending the full document each time.
"""

from __future__ import annotations

from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.config import Settings
from app.models import Answer


class QAService:
    """Wraps Llama Index to answer questions over a document using RAG."""

    def __init__(self, settings: Settings, document_text: str | None = None):
        """Initialize QA service with an optional document.
        
        Args:
            settings: Application settings (API keys, model names, etc.)
            document_text: Optional document text to index immediately.
        """
        self._settings = settings
        self._index = None
        
        # Set up LLM and embedding model
        self._llm = OpenAI(
            api_key=settings.openai_api_key or None,
            model=settings.openai_model,
        )
        self._embed_model = OpenAIEmbedding(
            api_key=settings.openai_api_key or None,
            model="text-embedding-3-small",
        )
        
        # If document text provided, build index immediately
        if document_text:
            self._build_index(document_text)

    def _build_index(self, document_text: str) -> None:
        """Build a vector index from document text."""
        if not document_text.strip():
            raise ValueError("Document text is empty.")
        
        # Create a Llama Index Document
        doc = Document(text=document_text)
        
        # Create vector store index (automatically chunks and embeds)
        self._index = VectorStoreIndex.from_documents(
            [doc],
            embed_model=self._embed_model,
            llm=self._llm,
            show_progress=False,
        )

    def set_document(self, document_text: str) -> None:
        """Set or update the document to be indexed."""
        self._build_index(document_text)

    def answer(
        self,
        questions: list[str],
    ) -> list[Answer]:
        """Answer ``questions`` over the indexed document.

        Args:
            questions: The questions to answer, in order.

        Returns:
            One :class:`Answer` per question, in the same order.
        """
        if self._index is None:
            raise RuntimeError(
                "No document has been indexed. Call set_document() first."
            )

        query_engine = self._index.as_query_engine(
            llm=self._llm,
            similarity_top_k=self._settings.top_k,
        )

        answers = []
        for question in questions:
            try:
                # Query the index with the question
                response = query_engine.query(
                    self._format_question_prompt(question)
                )
                answer_text = str(response)
                found = answer_text.lower() != self._settings.not_found_text.lower()
            except Exception as exc:
                answer_text = self._settings.not_found_text
                found = False

            answers.append(
                Answer(
                    question=question,
                    answer=answer_text,
                    found=found,
                )
            )

        return answers

    def _format_question_prompt(self, question: str) -> str:
        """Format a question with grounding instructions."""
        return (
            f"{question}\n\n"
            f"If the answer is not in the provided context, respond with exactly: "
            f"{self._settings.not_found_text!r}"
        )

