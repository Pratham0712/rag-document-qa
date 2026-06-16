import sys
sys.path.append(".")
from src.vectorstore.embedder import embed_text

vec = embed_text("test sentence")
print(f"Actual embedding dimension: {len(vec)}")