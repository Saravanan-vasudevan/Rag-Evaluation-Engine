import json

from src.rag_engine import evaluation
from tests.conftest import groq_response


def test_faithfulness_parses_json(monkeypatch):
    response = groq_response(json.dumps({"faithfulness": 1.0, "explanation": "Supported"}))
    client = type("Client", (), {"chat": type("Chat", (), {"completions": type(
        "Completions", (), {"create": lambda self, **kwargs: response}
    )()})()})()
    monkeypatch.setattr(evaluation, "Groq", lambda **kwargs: client)

    result = evaluation.score_faithfulness(
        "What is RAG?", "It uses retrieval.", [{"text": "RAG uses retrieval."}], "key"
    )
    assert result["faithfulness"] == 1.0


def test_precision_counts_relevant_chunks(monkeypatch):
    payload = {"judgements": [{"chunk": 1, "relevant": True, "reason": "Matches"}]}
    response = groq_response(json.dumps(payload))
    client = type("Client", (), {"chat": type("Chat", (), {"completions": type(
        "Completions", (), {"create": lambda self, **kwargs: response}
    )()})()})()
    monkeypatch.setattr(evaluation, "Groq", lambda **kwargs: client)
    result = evaluation.score_precision("question", [{"text": "answer"}], "key")
    assert result["precision"] == 1.0
