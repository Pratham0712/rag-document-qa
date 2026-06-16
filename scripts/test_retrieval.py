import sys
sys.path.append(".")

from src.vectorstore.store import VectorStore
from src.retrieval.retriever import mmr_search

store = VectorStore(dimension=3072)
store.load("faiss_index")

query = "What is the main topic of this document?"

print("=== Plain similarity search ===")
plain_results = store.search(query, top_k=5)
for i, (chunk, score) in enumerate(plain_results):
    print(f"{i+1}. (score: {score:.3f}) {chunk.text[:80]}...")

print("\n=== MMR re-ranked search ===")
mmr_results = mmr_search(store, query, top_k=5, fetch_k=15, lambda_param=0.5)
for i, (chunk, score) in enumerate(mmr_results):
    print(f"{i+1}. (score: {score:.3f}) {chunk.text[:80]}...")