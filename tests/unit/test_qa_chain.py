import sys
sys.path.append(".")
import pytest
from src.chain.prompts import build_context_block, build_grounded_prompt
from src.ingestion.chunker import Chunk


def make_chunks():
    return [
        (Chunk(chunk_id="c1", text="The sky is blue due to Rayleigh scattering.", source="science.pdf", page_number=3, strategy="fixed", chunk_index=0), 0.91),
        (Chunk(chunk_id="c2", text="Water boils at 100 degrees Celsius at sea level.", source="science.pdf", page_number=5, strategy="fixed", chunk_index=1), 0.85),
    ]


def test_context_block_includes_chunk_numbers():
    context = build_context_block(make_chunks())
    assert "[chunk 1]" in context
    assert "[chunk 2]" in context


def test_context_block_includes_source_metadata():
    context = build_context_block(make_chunks())
    assert "science.pdf" in context
    assert "page: 3" in context


def test_context_block_includes_actual_text():
    context = build_context_block(make_chunks())
    assert "Rayleigh scattering" in context
    assert "100 degrees Celsius" in context


def test_grounded_prompt_includes_question():
    prompt = build_grounded_prompt("Why is the sky blue?", make_chunks())
    assert "Why is the sky blue?" in prompt


def test_grounded_prompt_includes_grounding_rules():
    prompt = build_grounded_prompt("test question", make_chunks())
    assert "cannot find this information" in prompt
    assert "ONLY" in prompt


def test_grounded_prompt_empty_chunks():
    prompt = build_grounded_prompt("test question", [])
    assert "test question" in prompt