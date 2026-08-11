"""Text chunking strategies.

Two approaches: fixed-size token windows (simple, predictable) and
sentence-aware splitting (keeps chunks from cutting mid-sentence, which
tends to help retrieval precision on prose-heavy documents).
"""

import re
import tiktoken

_ENCODING = "cl100k_base"


def _encoder():
    return tiktoken.get_encoding(_ENCODING)


def chunk_fixed(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    enc = _encoder()
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


def chunk_semantic(text: str, target_size: int = 400) -> list[str]:
    enc = _encoder()
    # naive sentence split — good enough for the doc types this handles (reports,
    # articles, notes). Doesn't cope well with abbreviations like "Dr." but that's
    # an acceptable tradeoff over pulling in a full sentence tokenizer dependency.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_size = [], [], 0
    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))

        if sentence_tokens > target_size:
            # single sentence overflows the target on its own, flush what we have
            # and fall back to a hard split for this one
            if current:
                chunks.append(" ".join(current))
                current, current_size = [], 0
            chunks.extend(chunk_fixed(sentence, target_size, overlap=50))
            continue

        if current_size + sentence_tokens > target_size and current:
            chunks.append(" ".join(current))
            # carry the last sentence forward for a bit of continuity between chunks
            current = [current[-1]]
            current_size = len(enc.encode(current[0]))

        current.append(sentence)
        current_size += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks
