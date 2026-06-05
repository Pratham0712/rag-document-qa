\# RAG Document Q\&A



> Ask questions about any PDF. Get answers grounded strictly in the

> document — with source attribution and measurable hallucination rates.



\[!\[Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)

\[!\[LangChain](https://img.shields.io/badge/LangChain-0.3-purple)](https://langchain.com)

\[!\[Gemini](https://img.shields.io/badge/Gemini-2.5\_Flash-orange)](https://ai.google.dev)

\[!\[License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)



\---



\## The Problem



Large language models hallucinate. Ask ChatGPT about a specific internal

document, a legal contract, or a research paper it has never seen — it

will confidently make up answers. This is not acceptable in any serious

application.



\## The Solution



Retrieval Augmented Generation (RAG) forces the model to answer only

from retrieved document content. Every answer is traceable back to an

exact chunk of the source document. If the answer is not in the

document, the system says so.



\---



\## Architecture

PDF Document

│

▼

┌─────────────────┐

│   Ingestion     │  PyPDF2 + 3 chunking strategies

│   src/ingestion │  fixed / sentence-aware / semantic

└────────┬────────┘

│ chunks

▼

┌─────────────────┐

│  Vector Store   │  Gemini text-embedding-001

│  src/vectorstore│  FAISS index persisted to disk

└────────┬────────┘

│ embeddings

▼

┌─────────────────┐

│   Retrieval     │  Cosine similarity + MMR reranking

│  src/retrieval  │  metadata filtering, top-k search

└────────┬────────┘

│ relevant chunks

▼

┌─────────────────┐

│     Chain       │  LangChain RetrievalQA

│   src/chain     │  custom prompts, source attribution

└────────┬────────┘

│ grounded answer

▼

┌─────────────────┐

│      API        │  FastAPI — async, validated, documented

│    src/api      │  POST /upload  POST /query  GET /docs

└─────────────────┘



\---



\## What makes this different from a tutorial RAG



Most RAG tutorials stop at "it answers questions." This project goes

further in three ways.



\*\*Chunking benchmarks\*\* — three chunking strategies are implemented and

evaluated against each other. Fixed-size chunking, sentence-aware

chunking, and semantic chunking. The evaluation on Day 8 produces real

numbers showing which strategy reduces hallucination most.



\*\*RAGAS evaluation\*\* — every build is scored on faithfulness, answer

relevancy, and context recall using the RAGAS framework. The scores

are in this README. A recruiter can see exactly how well the system

performs.



\*\*Production patterns\*\* — the codebase uses dependency injection,

Pydantic validation, async FastAPI endpoints, Docker containerisation,

and GitHub Actions CI. It is built the way a real engineering team

would build it, not the way a tutorial teaches it.



\---



\## Tech Stack



| Layer | Technology | Why |

|---|---|---|

| LLM | Gemini 2.5 Flash | Free tier, fast, high quality |

| Embeddings | Gemini Embedding 001 | 768-dim, free, accurate |

| Vector store | FAISS | Zero infra, runs locally |

| RAG framework | LangChain 0.3 | Industry standard |

| API | FastAPI | Async, auto-docs, production ready |

| Evaluation | RAGAS | Industry standard RAG metrics |

| Containerisation | Docker | Reproducible anywhere |

| CI | GitHub Actions | Tests run on every push |



\---



\## Evaluation Results



Measured using RAGAS on a 20-question test set across 3 documents.



| Metric | Score |

|---|---|

| Faithfulness | Coming Day 8 |

| Answer relevancy | Coming Day 8 |

| Context recall | Coming Day 8 |

| Hallucination rate vs naive prompting | Coming Day 8 |



\---



\## Quick Start



\*\*Requirements:\*\* Python 3.11+, a free Gemini API key from

\[aistudio.google.com](https://aistudio.google.com)



```bash

git clone https://github.com/Pratham0712/rag-document-qa

cd rag-document-qa

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

copy .env.example .env

```



Open `.env` and add your Gemini API key. Then:



```bash

uvicorn src.api.main:app --reload

```



Open \[http://localhost:8000/docs](http://localhost:8000/docs) for the

interactive API documentation.



\---



\## Project Structure

rag-document-qa/

├── src/

│   ├── ingestion/       PDF loading, 3 chunking strategies

│   ├── vectorstore/     FAISS index, embedding pipeline

│   ├── retrieval/       similarity search, MMR reranking

│   ├── chain/           LangChain QA, prompt templates

│   └── api/             FastAPI endpoints, Pydantic models

├── tests/

│   ├── unit/            isolated function tests

│   └── integration/     full pipeline tests

├── eval/

│   ├── test\_data/       20-question evaluation dataset

│   └── reports/         RAGAS score outputs

├── docs/

│   └── adr/             architecture decision records

├── scripts/             dev utilities

├── Dockerfile           container definition

├── docker-compose.yml   local dev orchestration

├── .env.example         environment variable template

└── requirements.txt     pinned dependencies



\---



\## API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| POST | /upload | Upload a PDF document |

| POST | /query | Ask a question, get grounded answer |

| GET | /documents | List all uploaded documents |

| DELETE | /documents/{id} | Remove a document |

| GET | /health | Service health check |



\---



\## Architecture Decisions



All major technical decisions are documented in `docs/adr/` with

context, options considered, and reasoning.



\- \[ADR-001](docs/adr/ADR-001-chunking-strategy.md) — Chunking strategy

\- \[ADR-002](docs/adr/ADR-002-vector-store-choice.md) — Vector store selection







