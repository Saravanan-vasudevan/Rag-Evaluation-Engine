"""LLM-as-judge scoring for retrieval precision and answer faithfulness,
plus a small JSON log of past evaluation runs.

Judging is done with a cheaper Claude model since it's a secondary call
and doesn't need the same reasoning budget as the primary answer.
"""

import json
from pathlib import Path

import anthropic

from .config import EVAL_LOG_PATH, JUDGE_MODEL


def score_faithfulness(question: str, answer: str, chunks: list[dict], api_key: str) -> dict:
    context = "\n\n".join(chunk["text"][:400] for chunk in chunks)
    prompt = f"""Rate whether this answer is faithful to the context (0.0-1.0).
Faithful = only uses info from context. Unfaithful = introduces outside facts.

Context: {context}

Question: {question}
Answer: {answer}

Respond ONLY with JSON: {{"faithfulness": 0.85, "explanation": "brief reason"}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.content[0].text.strip())
    except Exception as exc:
        # judge call failing shouldn't take down the whole eval run — surface a
        # sentinel score so the UI can show "scoring failed" instead of crashing
        return {"faithfulness": -1.0, "explanation": f"Scoring failed: {exc}"}


def score_precision(question: str, chunks: list[dict], api_key: str) -> dict:
    chunk_list = "\n\n".join(
        f"Chunk {i + 1}: {chunk['text'][:250]}" for i, chunk in enumerate(chunks)
    )
    prompt = f"""For each chunk, is it relevant to the question?
Question: {question}
{chunk_list}

Respond ONLY with JSON array:
[{{"chunk": 1, "relevant": true, "reason": "brief"}}]"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        judgements = json.loads(response.content[0].text.strip())
        relevant_count = sum(1 for j in judgements if j.get("relevant", False))
        return {
            "precision": round(relevant_count / len(chunks), 3),
            "relevant_count": relevant_count,
            "judgements": judgements,
        }
    except Exception as exc:
        return {"precision": -1.0, "relevant_count": 0, "judgements": [], "error": str(exc)}


def load_eval_log() -> list[dict]:
    if not Path(EVAL_LOG_PATH).exists():
        return []
    try:
        with open(EVAL_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def save_eval_log(records: list[dict]) -> None:
    with open(EVAL_LOG_PATH, "w") as f:
        json.dump(records, f)
