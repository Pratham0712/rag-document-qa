# ADR-001: Chunking strategy for document ingestion

Date: 2025-06-06
Status: Accepted

## Context

RAG quality depends heavily on how documents are split into chunks.
Chunks that are too large dilute the relevant signal. Chunks that are
too small lose surrounding context. We need a strategy that balances
retrieval precision with answer completeness.

## Options considered

### Option 1 - Fixed-size chunking
Split every N tokens with an overlap of M tokens.
Pro: Simple, fast, deterministic.
Con: Cuts mid-sentence and mid-paragraph, destroys semantic coherence.

### Option 2 - Sentence-aware chunking
Split on sentence boundaries, group until token limit is reached.
Pro: Preserves sentence completeness.
Con: Chunks vary in size, dense paragraphs stay large.

### Option 3 - Semantic chunking
Embed each sentence, detect topic shifts via cosine distance.
Pro: Chunks align with actual ideas in the document.
Con: Slower because it requires embedding during ingestion.

## Decision

We implement all three and benchmark on Day 8 using RAGAS metrics.
Default for development is sentence-aware chunking as a baseline.
The ingestion module exposes a chunking_strategy parameter so the
strategy is swappable without changing any calling code.

## Consequences

Retrieval quality can be compared across all three strategies using
the same test set. This produces real benchmark numbers for the resume.
