import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

genai.configure(api_key=API_KEY)


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type=task_type
    )
    return result["embedding"]


def embed_query(query: str) -> list[float]:
    return embed_text(query, task_type="retrieval_query")


def embed_batch(texts: list[str], delay: float = 0.1) -> list[list[float]]:
    embeddings = []
    total = len(texts)

    for i, text in enumerate(texts):
        embedding = embed_text(text, task_type="retrieval_document")
        embeddings.append(embedding)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Embedded {i + 1}/{total} chunks")

        time.sleep(delay)

    return embeddings


def get_embedding_dimension() -> int:
    sample = embed_text("dimension check")
    return len(sample)