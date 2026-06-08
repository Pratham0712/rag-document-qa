import sys
sys.path.append(".")

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_document

PDF_PATH = "data/sample.pdf"

print("=" * 50)
print("TESTING LOADER")
print("=" * 50)
pages = load_pdf(PDF_PATH)
print(f"Pages loaded: {len(pages)}")
print(f"First page preview: {pages[0].text[:200]}\n")

print("=" * 50)
print("TESTING FIXED CHUNKING")
print("=" * 50)
fixed_chunks = chunk_document(pages, strategy="fixed")
print(f"Total chunks: {len(fixed_chunks)}")
print(f"First chunk: {fixed_chunks[0].text[:200]}\n")

print("=" * 50)
print("TESTING SENTENCE CHUNKING")
print("=" * 50)
sentence_chunks = chunk_document(pages, strategy="sentence")
print(f"Total chunks: {len(sentence_chunks)}")
print(f"First chunk: {sentence_chunks[0].text[:200]}\n")

print("=" * 50)
print("CHUNK SIZE COMPARISON")
print("=" * 50)
fixed_avg = sum(len(c.text.split()) for c in fixed_chunks) / len(fixed_chunks)
sentence_avg = sum(len(c.text.split()) for c in sentence_chunks) / len(sentence_chunks)
print(f"Fixed avg words per chunk:    {fixed_avg:.1f}")
print(f"Sentence avg words per chunk: {sentence_avg:.1f}")