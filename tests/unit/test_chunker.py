import sys
sys.path.append(".")
import pytest
from src.ingestion.loader import DocumentPage
from src.ingestion.chunker import (
    chunk_fixed,
    chunk_sentence,
    chunk_document,
    Chunk
)

SAMPLE_PAGE = DocumentPage(
    page_number=1,
    text="""Machine learning is a subset of artificial intelligence.
    It enables systems to learn from data automatically. Neural networks
    are inspired by the human brain. Deep learning uses multiple layers
    to extract features. Natural language processing helps computers
    understand human language. Computer vision allows machines to
    interpret images. Reinforcement learning trains agents through
    reward and punishment. Transfer learning reuses knowledge from
    one task to another.""",
    source="test.pdf"
)


def test_fixed_chunking_returns_chunks():
    chunks = chunk_fixed([SAMPLE_PAGE], chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_fixed_chunking_respects_size():
    chunks = chunk_fixed([SAMPLE_PAGE], chunk_size=20, overlap=0)
    for chunk in chunks[:-1]:
        word_count = len(chunk.text.split())
        assert word_count <= 20


def test_fixed_chunking_has_correct_metadata():
    chunks = chunk_fixed([SAMPLE_PAGE])
    assert chunks[0].source == "test.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].strategy == "fixed"


def test_sentence_chunking_returns_chunks():
    chunks = chunk_sentence([SAMPLE_PAGE])
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_sentence_chunking_has_correct_strategy():
    chunks = chunk_sentence([SAMPLE_PAGE])
    assert all(c.strategy == "sentence" for c in chunks)


def test_chunk_document_fixed():
    chunks = chunk_document([SAMPLE_PAGE], strategy="fixed")
    assert len(chunks) > 0
    assert chunks[0].strategy == "fixed"


def test_chunk_document_sentence():
    chunks = chunk_document([SAMPLE_PAGE], strategy="sentence")
    assert len(chunks) > 0
    assert chunks[0].strategy == "sentence"


def test_chunk_document_invalid_strategy():
    with pytest.raises(ValueError):
        chunk_document([SAMPLE_PAGE], strategy="invalid")


def test_chunk_ids_are_unique():
    chunks = chunk_document([SAMPLE_PAGE], strategy="fixed")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_empty_chunks():
    chunks = chunk_document([SAMPLE_PAGE], strategy="sentence")
    assert all(c.text.strip() != "" for c in chunks)