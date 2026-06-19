#!/usr/bin/env python3
"""Integration demo: test QAService with real OpenAI API and sample document.

This script demonstrates the full Llama Index RAG pipeline:
1. Load a document
2. Create embeddings and vector index
3. Answer questions using retrieved chunks

Requires:
- OPENAI_API_KEY set in .env
- sample_document.txt and sample_questions.json in the repo root

Usage:
    python demo_integration.py
"""

import json
import sys
from pathlib import Path

from app.config import Settings
from app.services.qa import QAService


def main():
    """Run the integration demo."""
    # Load settings (reads OPENAI_API_KEY from .env)
    print("Loading settings...")
    settings = Settings()
    
    if not settings.openai_api_key:
        print("❌ Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    
    print(f"✓ Using model: {settings.openai_model}")
    print(f"✓ Chunk size: {settings.chunk_size}")
    print(f"✓ Retrieve top-k: {settings.top_k}")
    print()
    
    # Load sample document
    doc_path = Path("sample_document.txt")
    if not doc_path.exists():
        print(f"❌ Error: {doc_path} not found")
        sys.exit(1)
    
    print(f"Loading document: {doc_path}")
    with open(doc_path) as f:
        document_text = f.read()
    print(f"✓ Document size: {len(document_text)} chars")
    print()
    
    # Initialize QA service and build index
    print("Building vector index (embedding document chunks)...")
    qa = QAService(settings, document_text=document_text)
    print("✓ Index built")
    print()
    
    # Load sample questions
    questions_path = Path("sample_questions.json")
    if not questions_path.exists():
        print(f"❌ Error: {questions_path} not found")
        sys.exit(1)
    
    print(f"Loading questions: {questions_path}")
    with open(questions_path) as f:
        questions = json.load(f)
    print(f"✓ Loaded {len(questions)} questions")
    print()
    
    # Answer questions
    print("=" * 70)
    print("ANSWERING QUESTIONS")
    print("=" * 70)
    print()
    
    try:
        answers = qa.answer(questions)
        
        for i, ans in enumerate(answers, 1):
            print(f"Q{i}: {ans.question}")
            print(f"A:  {ans.answer}")
            print(f"    Found in document: {'✓ Yes' if ans.found else '✗ No (not found)'}")
            print()
    
    except Exception as e:
        print(f"❌ Error during answering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
