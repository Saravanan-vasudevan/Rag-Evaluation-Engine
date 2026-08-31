"""LLM-as-judge scoring for retrieval precision and answer faithfulness,
plus a small JSON log of past evaluation runs.
"""

import json
from pathlib import Path

from groq import Groq

from .config import EVAL_LOG_PATH, JUDGE_MODEL


def score_faithfulness(question: str, answer: str, chunks: list[dict], api_key: str) -> dict:
    context = "\n\n".join(chunk["text"][:400] for chunk in chunks)
    prompt = f"""Rate whether this answer is faithful to the context (0.0-1.0).
Faithful = only uses info from context. Unfaithful = introduces outside facts.

Context: {context}

Question: {question}
Answer: {answer}

Respond ONLY with a valid JSON object: {{"faithfulness": 0.85, "explanation": "brief reason"}}"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as exc:
        return {"faithfulness": -1.0, "explanation": f"Scoring failed: {exc}"}


def score_precision(question: str, chunks: list[dict], api_key: str) -> dict:
    chunk_list = "\n\n".join(
        f"Chunk {i + 1}: {chunk['text'][:250]}" for i, chunk in enumerate(chunks)
    )
    prompt = f"""For each chunk, is it relevant to the question?
Question: {question}
{chunk_list}

Respond ONLY with a valid JSON object containing a "judgements" array:
{{"judgements": [{{"chunk": 1, "relevant": true, "reason": "brief"}}]}}"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = json.loads(response.choices[0].message.content.strip())
        judgements = content.get("judgements", [])
        relevant_count = sum(1 for j in judgements if j.get("relevant", False))
        return {
            "precision": round(relevant_count / len(chunks), 3) if chunks else 0.0,
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
