"""FastAPI application exposing the document Q&A bot.

Endpoints:
    GET  /health        — liveness check.
    POST /ask           — answer questions over a plain-text document (JSON).
    POST /ask/upload    — answer questions over an uploaded PDF or text file
                          plus an uploaded/inline questions list (multipart).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pypdf import PdfReader

from app.config import Settings, get_settings
from app.models import AskRequest, AskResponse
from app.services.qa import QAService
from app.services.question_loader import QuestionLoadError, parse_questions

app = FastAPI(
    title="Zania Q&A Bot",
    description="Answer questions grounded in an uploaded document.",
    version="1.0.0",
)


@lru_cache
def get_qa_service() -> QAService:
    """Build (and cache) the QA service. Cached so the OpenAI client and
    its connection pool are reused across requests."""
    return QAService(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    settings: Settings = Depends(get_settings),
    qa: QAService = Depends(get_qa_service),
) -> AskResponse:
    """Answer questions over a plain-text document supplied in the JSON body."""
    try:
        questions = parse_questions(
            _to_json(req.questions), max_questions=settings.max_questions
        )
    except QuestionLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        if not req.document.strip():
            raise ValueError("Document text is empty.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        qa.set_document(req.document)
        answers = qa.answer(questions)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error: {exc}") from exc

    return AskResponse(answers=answers)


@app.post("/ask/upload", response_model=AskResponse)
async def ask_upload(
    document: UploadFile = File(..., description="PDF or text document."),
    questions_file: UploadFile | None = File(
        default=None, description="JSON file with the questions."
    ),
    questions: str | None = Form(
        default=None, description="Inline JSON array of questions."
    ),
    settings: Settings = Depends(get_settings),
    qa: QAService = Depends(get_qa_service),
) -> AskResponse:
    """Answer questions over an uploaded PDF/text document.

    Provide the questions either as an uploaded JSON file (``questions_file``)
    or inline as a JSON string field (``questions``).
    """
    raw_questions = await _read_questions_input(questions_file, questions)
    try:
        parsed_questions = parse_questions(
            raw_questions, max_questions=settings.max_questions
        )
    except QuestionLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    data = await document.read()
    if len(data) > settings.max_document_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Document exceeds {settings.max_document_bytes} bytes.",
        )

    try:
        document_text = _extract_text(
            data,
            filename=document.filename,
            media_type=document.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        qa.set_document(document_text)
        answers = qa.answer(parsed_questions)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error: {exc}") from exc

    return AskResponse(answers=answers)


async def _read_questions_input(
    questions_file: UploadFile | None, questions: str | None
) -> bytes | str:
    if questions_file is not None:
        return await questions_file.read()
    if questions is not None:
        return questions
    raise HTTPException(
        status_code=422,
        detail="Provide questions via 'questions_file' or the 'questions' field.",
    )


def _extract_text(
    data: bytes,
    filename: str | None = None,
    media_type: str | None = None,
) -> str:
    """Extract text from uploaded document (PDF or text)."""
    if not data:
        raise ValueError("Document is empty.")

    # Detect PDF by media type, filename, or magic header
    is_pdf = (
        (media_type == "application/pdf")
        or (filename or "").lower().endswith(".pdf")
        or data[:5] == b"%PDF-"
    )

    if is_pdf:
        try:
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(data))
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
            if not text.strip():
                raise ValueError("PDF contains no extractable text.")
            return text
        except Exception as exc:
            raise ValueError(f"Failed to extract text from PDF: {exc}") from exc

    # Treat as UTF-8 text
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Document is not a PDF and could not be decoded as UTF-8 text."
        ) from exc

    if not text.strip():
        raise ValueError("Document text is empty.")

    return text


def _to_json(items: list[str]) -> str:
    import json

    return json.dumps(items)

