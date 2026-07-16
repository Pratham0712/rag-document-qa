import sys
sys.path.append(".")

from src.vectorstore.store import VectorStore
from src.chain.qa_chain import answer_question

store = VectorStore(dimension=3072)
store.load("faiss_index")

questions = [
    "What is the main contribution of this paper?",
    "What security mechanisms are described?",
    "What is the capital of France?"
]

for question in questions:
    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)

    result = answer_question(store, question, top_k=4)

    print(f"\nA: {result.answer}\n")
    print("Sources used:")
    for src in result.sources:
        print(f"  chunk {src['chunk_number']}: {src['source']} page {src['page_number']} score {src['relevance_score']}")
    print()
