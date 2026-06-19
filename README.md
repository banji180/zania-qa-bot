# Zania Q&A Bot

A FastAPI service that answers questions grounded in an uploaded document using **Llama Index** for cost-efficient vector retrieval and **OpenAI's gpt-4o-mini** for LLM inference.

## How It Works

Instead of sending entire documents to the LLM (expensive), this service uses **Retrieval-Augmented Generation (RAG)**:

1. **Chunk & Embed**: Document is split into overlapping chunks and embedded using `text-embedding-3-small`
2. **Index**: Chunks are stored in a vector index
3. **Retrieve**: For each question, relevant chunks are retrieved via similarity search
4. **Answer**: Only retrieved chunks (not the full document) are sent to gpt-4o-mini

This reduces token usage by 10-100x compared to sending full documents, drastically lowering costs.

## Architecture

```
app/
  main.py                  FastAPI app + endpoints
  config.py                Settings (env / .env)
  models.py                Pydantic request / response models
  services/
    qa.py                  Llama Index RAG service
    question_loader.py     Parse + normalise the questions list
tests/
  test_api.py              API tests with a stubbed QA service (no network)
  test_question_loader.py  Unit tests for question parsing
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

The interactive docs are at http://127.0.0.1:8000/docs.

## Usage

### `POST /ask` — plain-text document (JSON)

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{
        "document": "Acme Corp was founded in 2019 by Dana Lee.",
        "questions": ["When was Acme founded?", "What is its revenue?"]
      }'
```

```json
{
  "answers": [
    {"question": "When was Acme founded?", "answer": "2019", "found": true},
    {"question": "What is its revenue?", "answer": "Data Not Available", "found": false}
  ]
}
```

### `POST /ask/upload` — PDF or text upload (multipart)

Questions inline:

```bash
curl -s http://127.0.0.1:8000/ask/upload \
  -F 'document=@handbook.pdf;type=application/pdf' \
  -F 'questions=["What is the leave policy?", "Who approves expenses?"]'
```

Or questions as a JSON file (array, or `{"questions": [...]}`):

```bash
curl -s http://127.0.0.1:8000/ask/upload \
  -F 'document=@handbook.pdf;type=application/pdf' \
  -F 'questions_file=@questions.json;type=application/json'
```

## Configuration

All settings have defaults and can be overridden via environment variables or `.env` (see [.env.example](.env.example)):

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model id. |
| `CHUNK_SIZE` | `512` | Document chunk size (tokens). |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks. |
| `TOP_K` | `3` | Number of chunks to retrieve per question. |
| `NOT_FOUND_TEXT` | `Data Not Available` | Returned when an answer isn't in the document. |
| `MAX_QUESTIONS` | `50` | Per-request question cap. |
| `MAX_DOCUMENT_BYTES` | `33554432` | Upload size cap (32 MB). |

## Tests

```bash
pytest
```

The tests stub the QA service, so they run without an API key and make no network calls.

## Cost Efficiency

Using Llama Index RAG with gpt-4o-mini:
- **Typical doc**: Send ~1-3 chunks (500 tokens) instead of full 10-50KB docs
- **Cost per question**: ~0.01¢ vs ~0.10¢+ with full-doc approach
- **99% token savings** on repeated questions over large documents

