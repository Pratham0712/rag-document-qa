import os
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

load_dotenv()

from src.ingestion.loader import DocumentPage


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page_number: int
    strategy: str
    chunk_index: int


def chunk_fixed(
    pages: list[DocumentPage],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[Chunk]:
    chunks = []
    chunk_index = 0

    for page in pages:
        words = page.text.split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{page.source}_p{page.page_number}_c{chunk_index}",
                    text=chunk_text,
                    source=page.source,
                    page_number=page.page_number,
                    strategy="fixed",
                    chunk_index=chunk_index
                ))
                chunk_index += 1

            start += chunk_size - overlap

    print(f"Fixed chunking: {len(chunks)} chunks created")
    return chunks


def chunk_sentence(
    pages: list[DocumentPage],
    max_chunk_size: int = 500,
    overlap_sentences: int = 1
) -> list[Chunk]:
    import re
    chunks = []
    chunk_index = 0

    for page in pages:
        sentences = re.split(r'(?<=[.!?])\s+', page.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk_sentences = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            if current_size + sentence_size > max_chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{page.source}_p{page.page_number}_c{chunk_index}",
                    text=chunk_text,
                    source=page.source,
                    page_number=page.page_number,
                    strategy="sentence",
                    chunk_index=chunk_index
                ))
                chunk_index += 1

                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_size = sum(len(s.split()) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_size += sentence_size

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Chunk(
                chunk_id=f"{page.source}_p{page.page_number}_c{chunk_index}",
                text=chunk_text,
                source=page.source,
                page_number=page.page_number,
                strategy="sentence",
                chunk_index=chunk_index
            ))
            chunk_index += 1

    print(f"Sentence chunking: {len(chunks)} chunks created")
    return chunks


def chunk_semantic(
    pages: list[DocumentPage],
    threshold: float = 0.3
) -> list[Chunk]:
    import re

    api_key = os.getenv("GOOGLE_API_KEY")
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    genai.configure(api_key=api_key)

    all_sentences = []
    sentence_metadata = []

    for page in pages:
        sentences = re.split(r'(?<=[.!?])\s+', page.text.strip())
        for s in sentences:
            s = s.strip()
            if s:
                all_sentences.append(s)
                sentence_metadata.append({
                    "source": page.source,
                    "page_number": page.page_number
                })

    print(f"Embedding {len(all_sentences)} sentences for semantic chunking...")

    embeddings = []
    for sentence in all_sentences:
        result = genai.embed_content(
            model=embed_model,
            content=sentence,
            task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])

    embeddings = np.array(embeddings)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    split_points = [0]
    for i in range(1, len(embeddings)):
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
        if similarity < (1 - threshold):
            split_points.append(i)
    split_points.append(len(all_sentences))

    chunks = []
    for chunk_index, start in enumerate(split_points[:-1]):
        end = split_points[chunk_index + 1]
        chunk_sentences = all_sentences[start:end]
        chunk_text = " ".join(chunk_sentences)
        meta = sentence_metadata[start]

        if chunk_text.strip():
            chunks.append(Chunk(
                chunk_id=f"{meta['source']}_semantic_c{chunk_index}",
                text=chunk_text,
                source=meta["source"],
                page_number=meta["page_number"],
                strategy="semantic",
                chunk_index=chunk_index
            ))

    print(f"Semantic chunking: {len(chunks)} chunks created")
    return chunks


def chunk_document(
    pages: list[DocumentPage],
    strategy: str = "sentence",
    **kwargs
) -> list[Chunk]:
    strategies = {
        "fixed": chunk_fixed,
        "sentence": chunk_sentence,
        "semantic": chunk_semantic
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")

    return strategies[strategy](pages, **kwargs)