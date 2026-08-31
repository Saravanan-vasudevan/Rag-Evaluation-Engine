import sys
from types import ModuleType

vectorstore = ModuleType("src.rag_engine.vectorstore")
vectorstore.retrieve = lambda question, top_k: []
sys.modules.setdefault("src.rag_engine.vectorstore", vectorstore)

from src.rag_engine import qa


def test_answer_without_documents(monkeypatch):
    monkeypatch.setattr(qa, "retrieve", lambda question, top_k: [])
    result = qa.answer_question("Anything indexed?", 3, "unused")
    assert result["chunks_used"] == 0
    assert result["sources"] == []
