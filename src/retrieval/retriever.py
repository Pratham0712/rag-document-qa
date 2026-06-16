import numpy as np
from src.vectorstore.store import VectorStore
from src.vectorstore.embedder import embed_query, embed_text
from src.ingestion.chunker import Chunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def mmr_search(
    store: VectorStore,
    query: str,
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_param: float = 0.5,
    source_filter: str = None
) -> list[tuple[Chunk, float]]:
    if len(store) == 0:
        raise ValueError("Vector store is empty.")

    candidates = store.search(query, top_k=min(fetch_k, len(store)))

    if source_filter:
        candidates = [(c, s) for c, s in candidates if c.source == source_filter]

    if not candidates:
        return []

    query_vector = np.array(embed_query(query))

    candidate_vectors = []
    for chunk, _ in candidates:
        vec = np.array(embed_text(chunk.text))
        candidate_vectors.append(vec)

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    relevance_scores = [
        cosine_similarity(query_vector, vec) for vec in candidate_vectors
    ]

    first_pick = int(np.argmax(relevance_scores))
    selected_indices.append(first_pick)
    remaining_indices.remove(first_pick)

    while len(selected_indices) < min(top_k, len(candidates)) and remaining_indices:
        mmr_scores = []

        for idx in remaining_indices:
            relevance = relevance_scores[idx]

            max_similarity_to_selected = max(
                cosine_similarity(candidate_vectors[idx], candidate_vectors[sel])
                for sel in selected_indices
            )

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_similarity_to_selected
            )
            mmr_scores.append((idx, mmr_score))

        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    results = [
        (candidates[idx][0], relevance_scores[idx])
        for idx in selected_indices
    ]

    return results


def filter_by_source(
    results: list[tuple[Chunk, float]],
    source: str
) -> list[tuple[Chunk, float]]:
    return [(c, s) for c, s in results if c.source == source]


def filter_by_min_score(
    results: list[tuple[Chunk, float]],
    min_score: float = 0.5
) -> list[tuple[Chunk, float]]:
    return [(c, s) for c, s in results if s >= min_score]
