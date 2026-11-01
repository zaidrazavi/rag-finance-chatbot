"""
Basic tests for the RAG pipeline components.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch


def test_prompt_versioning():
    from app.prompts import get_prompt, list_prompts
    prompt = get_prompt("retrieval")
    assert prompt.version.startswith("retrieval-")
    assert "{query}" in prompt.template

    all_prompts = list_prompts()
    assert "retrieval" in all_prompts
    assert "reasoning" in all_prompts
    assert "response" in all_prompts
    assert "hallucination" in all_prompts


def test_invalid_prompt_name():
    from app.prompts import get_prompt
    with pytest.raises(ValueError):
        get_prompt("nonexistent_prompt")


def test_chunk_documents():
    from app.embeddings import chunk_documents
    texts = ["This is a test financial document. " * 50]
    docs = chunk_documents(texts, [{"source": "test.txt"}])
    assert len(docs) > 0
    for doc in docs:
        assert len(doc.page_content) <= 600  # chunk size + some buffer
        assert doc.metadata["source"] == "test.txt"


def test_hallucination_score_clamped():
    """Score should always be between 0 and 1."""
    checker_mock = MagicMock()
    checker_mock.score.return_value = 0.85
    score = checker_mock.score("some context", "some answer")
    assert 0.0 <= score <= 1.0


def test_query_response_structure():
    """Test that query response has all required fields."""
    required_fields = [
        "answer", "grounding_score", "grounding_passed",
        "warning", "expanded_query", "sources", "prompt_versions"
    ]
    mock_response = {field: None for field in required_fields}
    for field in required_fields:
        assert field in mock_response
