# ADR-002: Vector store selection

Date: 2025-06-06
Status: Accepted

## Context

We need a vector store to persist and query document embeddings.
The choice affects query latency, memory usage, and deployment
complexity. Simplicity and reproducibility are the top priorities.

## Options considered

### Option 1 - FAISS
In-process library, no server required, file-based persistence.
Pro: Zero infrastructure, extremely fast, perfect for single machine.
Con: No built-in multi-tenancy, limited to available RAM.

### Option 2 - Chroma
Embeddable database with optional REST server mode.
Pro: Richer metadata filtering, persistent by default.
Con: Adds server dependency, more complex setup.

### Option 3 - Pinecone or Weaviate
Fully managed, horizontally scalable, production grade.
Pro: Real production option used at scale.
Con: Requires paid account, adds network dependency.

## Decision

FAISS. The goal is a fully reproducible demo that an interviewer
can clone and run in under 5 minutes with no external services.
FAISS achieves this with a single pip install and no running servers.

## Consequences

FAISS index is persisted to disk after every ingestion run.
The vectorstore module abstracts FAISS behind an interface so
swapping to Chroma or Pinecone later requires changing one file only.
