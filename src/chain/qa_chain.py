import os
from dotenv import load_dotenv
import google.generativeai as genai
from dataclasses import dataclass

from src.vectorstore.store import VectorStore
from src.retrieval.retriever import mmr_search
from src.chain.prompts import build_grounded_prompt

load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


@dataclass
class QAResult:
    answer: str
    sources: list[dict]
    question: str


def answer_question(
    store: VectorStore,
    question: str,
    top_k: int = 5,
    fetch_k: int = 15,
    use_mmr: bool = True,
    source_filter: str = None
) -> QAResult:
    if use_mmr:
        results = mmr_search(
            store,
            question,
            top_k=top_k,
            fetch_k=fetch_k,
            source_filter=source_filter
        )
    else:
        results = store.search(question, top_k=top_k)
        if source_filter:
            results = [(c, s) for c, s in results if c.source == source_filter]

    if not results:
        return QAResult(
            answer="I cannot find this information in the provided documents.",
            sources=[],
            question=question
        )

    prompt = build_grounded_prompt(question, results)
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)

    sources = [
        {
            "chunk_number": i + 1,
            "source": chunk.source,
            "page_number": chunk.page_number,
            "relevance_score": round(score, 4),
            "preview": chunk.text[:150]
        }
        for i, (chunk, score) in enumerate(results)
    ]

    return QAResult(
        answer=response.text.strip(),
        sources=sources,
        question=question
    )