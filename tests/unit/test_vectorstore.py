import sys
sys.path.append(".")
import pytest
import shutil
import os
from src.vectorstore.store import VectorStore
from src.ingestion.chunker import Chunk


def make_test_chunks():
    return [
        Chunk(chunk_id="c1", text="Cats are small domestic animals.", source="t.pdf", page_number=1, strategy="fixed", chunk_index=0),
        Chunk(chunk_id="c2", text="Dogs are loyal household pets.", source="t.pdf", page_number=1, strategy="fixed", chunk_index=1),
        Chunk(chunk_id="c3", text="Python is a popular programming language.", source="t.pdf", page_number=2, strategy="fixed", chunk_index=2),
    ]


def test_add_chunks_increases_count():
    store = VectorStore(dimension=3072)
    chunks = make_test_chunks()
    store.add_chunks(chunks)
    assert len(store) == 3


def test_search_returns_results():
    store = VectorStore(dimension=3072)
    store.add_chunks(make_test_chunks())
    results = store.search("Tell me about pets", top_k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)


def test_search_ranks_relevant_first():
    store = VectorStore(dimension=3072)
    store.add_chunks(make_test_chunks())
    results = store.search("programming languages", top_k=1)
    assert "Python" in results[0][0].text


def test_search_empty_store_raises():
    store = VectorStore(dimension=3072)
    with pytest.raises(ValueError):
        store.search("anything", top_k=3)


def test_save_and_load_roundtrip():
    store = VectorStore(dimension=3072)
    store.add_chunks(make_test_chunks())
    store.save("test_index_tmp")

    new_store = VectorStore(dimension=3072)
    new_store.load("test_index_tmp")

    assert len(new_store) == len(store)
    assert new_store.chunks[0].text == store.chunks[0].text

    shutil.rmtree("test_index_tmp")