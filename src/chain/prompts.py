GROUNDED_QA_PROMPT = """You are a precise document question-answering assistant.
You must answer ONLY using the information in the CONTEXT below.

Rules you must follow strictly:
1. Use ONLY facts present in the CONTEXT. Do not use any outside knowledge.
2. If the CONTEXT does not contain enough information to answer the question,
   respond exactly with: "I cannot find this information in the provided documents."
3. Do not guess, assume, or fabricate any detail not explicitly stated in the CONTEXT.
4. When you answer, mention which source chunk numbers support your answer,
   like this: (Source: chunk 2, chunk 4).
5. Keep your answer concise and directly address the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_context_block(chunks_with_scores: list) -> str:
    context_parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores):
        context_parts.append(
            f"[chunk {i + 1}] (source: {chunk.source}, page: {chunk.page_number})\n{chunk.text}"
        )
    return "\n\n".join(context_parts)


def build_grounded_prompt(question: str, chunks_with_scores: list) -> str:
    context = build_context_block(chunks_with_scores)
    return GROUNDED_QA_PROMPT.format(context=context, question=question)