"""Question answering: retrieve context, call Groq, return a structured answer."""

import time

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import ANSWER_MODEL, SYSTEM_PROMPT
from .vectorstore import retrieve


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def call_llm(prompt: str, api_key: str) -> str:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_question(question: str, top_k: int, api_key: str) -> dict:
    started_at = time.monotonic()
    chunks = retrieve(question, top_k)

    if not chunks:
        return {
            "answer": "No documents indexed yet. Upload some documents first.",
            "sources": [],
            "latency_ms": 0,
            "chunks_used": 0,
        }

    context_block = "\n\n---\n\n".join(
        f"[Passage {i + 1} — {chunk['source']}]\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    )
    prompt = f"Context:\n\n{context_block}\n\n---\n\nQuestion: {question}\n\nAnswer:"

    answer_text = call_llm(prompt, api_key)

    return {
        "answer": answer_text,
        "sources": chunks,
        "latency_ms": int((time.monotonic() - started_at) * 1000),
        "chunks_used": len(chunks),
    }
