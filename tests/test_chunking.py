import pytest

from src.rag_engine.chunking import chunk_fixed, chunk_sentence_aware


def test_chunk_fixed_respects_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_fixed(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # every chunk after the first should share some tail/head content with its
    # neighbour, since that's the whole point of the overlap
    first_tail = chunks[0].split()[-10:]
    assert all(word in chunks[1] for word in first_tail)


def test_chunk_semantic_keeps_sentences_whole():
    text = "First sentence here. Second sentence follows. Third one wraps it up."
    chunks = chunk_sentence_aware(text, target_size=400)

    # target size is generous relative to this text, so it should all land in one chunk
    assert len(chunks) == 1
    assert chunks[0].count(".") == 3


def test_chunk_semantic_handles_empty_string():
    assert chunk_sentence_aware("") == []


def test_chunk_fixed_handles_short_text():
    chunks = chunk_fixed("just a few words", chunk_size=400, overlap=80)
    assert len(chunks) == 1


def test_chunk_fixed_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_fixed("some text", chunk_size=100, overlap=100)
