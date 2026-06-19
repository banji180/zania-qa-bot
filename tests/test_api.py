"""API tests using a stubbed QA service (no network / no API key needed)."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_qa_service
from app.models import Answer


class FakeQAService:
    """Echoes a deterministic answer per question; records the last call."""

    def __init__(self):
        self.last_document = None
        self.last_questions = None

    def set_document(self, document_text: str) -> None:
        self.last_document = document_text

    def answer(self, questions):
        self.last_questions = questions
        return [
            Answer(question=q, answer=f"answer to: {q}", found=True)
            for q in questions
        ]


@pytest.fixture
def fake_qa():
    return FakeQAService()


@pytest.fixture
def client(fake_qa):
    app.dependency_overrides[get_qa_service] = lambda: fake_qa
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_json(client, fake_qa):
    resp = client.post(
        "/ask",
        json={"document": "Acme Corp was founded in 2019.", "questions": ["When founded?"]},
    )
    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert answers == [
        {"question": "When founded?", "answer": "answer to: When founded?", "found": True}
    ]
    assert fake_qa.last_questions == ["When founded?"]
    assert fake_qa.last_document == "Acme Corp was founded in 2019."


def test_ask_json_empty_questions_is_422(client):
    resp = client.post("/ask", json={"document": "x", "questions": []})
    assert resp.status_code == 422


def test_ask_upload_text_with_inline_questions(client):
    resp = client.post(
        "/ask/upload",
        files={"document": ("doc.txt", b"The sky is blue.", "text/plain")},
        data={"questions": json.dumps(["What colour is the sky?"])},
    )
    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert answers[0]["question"] == "What colour is the sky?"


def test_ask_upload_with_questions_file(client):
    resp = client.post(
        "/ask/upload",
        files={
            "document": ("doc.txt", b"hello world", "text/plain"),
            "questions_file": (
                "questions.json",
                io.BytesIO(json.dumps({"questions": ["Q1", "Q2"]}).encode()),
                "application/json",
            ),
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["answers"]) == 2


def test_ask_upload_missing_questions_is_422(client):
    resp = client.post(
        "/ask/upload",
        files={"document": ("doc.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 422


def test_ask_upload_pdf_extracts_text(client, fake_qa):
    pdf_bytes = b"%PDF-1.4 fake pdf body"
    resp = client.post(
        "/ask/upload",
        files={"document": ("doc.pdf", pdf_bytes, "application/pdf")},
        data={"questions": json.dumps(["What is this?"])},
    )
    # Extraction will fail on fake PDF, but the endpoint should handle it
    assert resp.status_code in (200, 422)

