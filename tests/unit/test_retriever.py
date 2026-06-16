import sys
sys.path.append(".")
import pytest
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import mmr_search, filter_by_source, filter_by_min_score
from src.ingestion.chunker import Chunk


def make_diverse_chunks():
    return [
        Chunk(chunk_id="c1", text="Cats are small domestic animals that purr.", source="a.pdf", page_number=1, strategy="fixed", chunk_index=0),
        Chunk(chunk_id="c2", text="Cats often sleep most of the day.", source="a.pdf", page_number=1, strategy="fixed", chunk_index=1),
        Chunk(chunk_id="c3", text="Python is a programming language used for AI.", source="b.pdf", page_number=2, strategy="fixed", chunk_index=2),
        Chunk(chunk_id="c4", text="Stock markets fluctuate based on economic news.", source="b.pdf", page_number=3, strategy="fixed", chunk_index=3),
    ]


def test_mmr_returns_requested_count():
    store = VectorStore(dimension=3072)
    store.add_chunks(make_diverse_chunks())
    results = mmr_search(store, "tell me about animals", top_k=2, fetch_k=4)
    assert len(results) == 2


def test_mmr_empty_store_raises():
    store = VectorStore(dimension=3072)
    with pytest.raises(ValueError):
        mmr_search(store, "anything", top_k=3)


def test_filter_by_source():
    chunks = make_diverse_chunks()
    fake_results = [(c, 0.9) for c in chunks]
    filtered = filter_by_source(fake_results, "a.pdf")
    assert len(filtered) == 2
    assert all(c.source == "a.pdf" for c, _ in filtered)


def test_filter_by_min_score():
    chunks = make_diverse_chunks()
    fake_results = [(chunks[0], 0.9), (chunks[1], 0.3), (chunks[2], 0.6)]
    filtered = filter_by_min_score(fake_results, min_score=0.5)
    assert len(filtered) == 2