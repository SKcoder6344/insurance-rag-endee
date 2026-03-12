"""
Unit tests for RAG pipeline components.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch


# --- Test: Text chunking ---

def test_chunk_text_basic():
    from app.config import settings
    settings.chunk_size = 5
    settings.chunk_overlap = 1

    from app.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.__new__(RAGPipeline)
    text = "one two three four five six seven eight"
    chunks = pipeline._chunk_text(text)
    assert len(chunks) >= 2
    assert "one" in chunks[0]


def test_chunk_text_short_document():
    from app.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.__new__(RAGPipeline)
    text = "short text"
    chunks = pipeline._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


# --- Test: Schemas ---

def test_query_request_validation():
    from app.schemas import QueryRequest
    req = QueryRequest(question="What is covered?", top_k=3)
    assert req.top_k == 3
    assert "covered" in req.question


def test_query_request_default_top_k():
    from app.schemas import QueryRequest
    req = QueryRequest(question="test question")
    assert req.top_k == 3


def test_query_request_invalid_too_short():
    from app.schemas import QueryRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QueryRequest(question="ab")


# --- Test: RAG answer with mocked Endee and OpenAI ---

@patch("app.rag_pipeline.OpenAI")
@patch("app.rag_pipeline.EndeeVectorStore")
def test_answer_returns_expected_structure(MockStore, MockOpenAI):
    from app.rag_pipeline import RAGPipeline

    # Mock embeddings
    mock_openai_instance = MagicMock()
    mock_openai_instance.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    mock_openai_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test answer about insurance coverage."))]
    )
    MockOpenAI.return_value = mock_openai_instance

    # Mock Endee store
    mock_store_instance = MagicMock()
    mock_store_instance.query.return_value = [
        {"text": "Health insurance covers hospitalization.", "source": "document_1", "similarity": 0.92}
    ]
    MockStore.return_value = mock_store_instance

    pipeline = RAGPipeline()
    result = pipeline.answer("What does health insurance cover?", top_k=3)

    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["similarity"] == 0.92


@patch("app.rag_pipeline.OpenAI")
@patch("app.rag_pipeline.EndeeVectorStore")
def test_answer_no_results(MockStore, MockOpenAI):
    from app.rag_pipeline import RAGPipeline

    mock_openai_instance = MagicMock()
    mock_openai_instance.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    MockOpenAI.return_value = mock_openai_instance

    mock_store_instance = MagicMock()
    mock_store_instance.query.return_value = []
    MockStore.return_value = mock_store_instance

    pipeline = RAGPipeline()
    result = pipeline.answer("random question")

    assert "No relevant" in result["answer"]
    assert result["sources"] == []
