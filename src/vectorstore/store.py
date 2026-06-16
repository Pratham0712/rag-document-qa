import os
import pickle
import numpy as np
import faiss

from src.vectorstore.embedder import embed_batch, embed_query
from src.ingestion.chunker import Chunk


class VectorStore:
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            print("No chunks to add")
            return

        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = embed_batch(texts)

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

        print(f"Added {len(chunks)} chunks. Total in store: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")

        query_vector = np.array([embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Vector store saved to {path}")

    def load(self, path: str):
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Vector store loaded from {path}. {len(self.chunks)} chunks.")

    def __len__(self):
        return len(self.chunks)