import sys
sys.path.append(".")

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_document
from src.vectorstore.store import VectorStore

PDF_PATH = "data/sample.pdf"

print("Loading PDF...")
pages = load_pdf(PDF_PATH)

print("\nChunking...")
chunks = chunk_document(pages, strategy="sentence")
print(f"Created {len(chunks)} chunks")

print("\nBuilding vector store...")
store = VectorStore(dimension=3072)
store.add_chunks(chunks)

print("\nSaving to disk...")
store.save("faiss_index")

print("\nTesting search...")
query = "What is the main topic of this document?"
results = store.search(query, top_k=3)

print(f"\nTop {len(results)} results for: '{query}'\n")
for i, (chunk, score) in enumerate(results):
    print(f"--- Result {i+1} (score: {score:.4f}) ---")
    print(f"Source: {chunk.source}, Page: {chunk.page_number}")
    print(f"Text: {chunk.text[:200]}...")
    print()