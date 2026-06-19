"""Tests for the question loader/normaliser."""

from __future__ import annotations

import pytest

from app.services.question_loader import QuestionLoadError, parse_questions


def test_parse_array_form():
    qs = parse_questions('["What is the cap table?", "Who is the CEO?"]', max_questions=50)
    assert qs == ["What is the cap table?", "Who is the CEO?"]


def test_parse_object_form():
    qs = parse_questions('{"questions": ["A", "B"]}', max_questions=50)
    assert qs == ["A", "B"]


def test_strips_and_dedupes_preserving_order():
    qs = parse_questions('["  A  ", "B", "A", "", "C"]', max_questions=50)
    assert qs == ["A", "B", "C"]


def test_accepts_bytes():
    qs = parse_questions(b'["A"]', max_questions=50)
    assert qs == ["A"]


def test_invalid_json_raises():
    with pytest.raises(QuestionLoadError):
        parse_questions("not json", max_questions=50)


def test_wrong_shape_raises():
    with pytest.raises(QuestionLoadError):
        parse_questions('"just a string"', max_questions=50)


def test_non_string_item_raises():
    with pytest.raises(QuestionLoadError):
        parse_questions("[1, 2, 3]", max_questions=50)


def test_empty_raises():
    with pytest.raises(QuestionLoadError):
        parse_questions("[]", max_questions=50)


def test_too_many_raises():
    payload = "[" + ",".join(f'"q{i}"' for i in range(5)) + "]"
    with pytest.raises(QuestionLoadError):
        parse_questions(payload, max_questions=3)
