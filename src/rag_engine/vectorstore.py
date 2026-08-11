"""ChromaDB collection management, ingestion and retrieval."""

import uuid
from pathlib import Path

import chromadb
import streamlit as st

from .config import CHROMA_DIR, COLLECTION_NAME
from .chunking import chunk_fixed, chunk_semantic
from .embeddings import embed_texts, embed_query


@st.cache_resource(show_spinner=False)
def get_collection():
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection(COLLECTION_NAME)
    get_collection.clear()  # force the cached resource to rebuild on next access


def ingest_document(text: str, filename: str, strategy: str = "semantic", chunk_size: int = 400) -> int:
    """Chunk, embed and store a document. Returns the number of chunks written."""
    if strategy == "semantic":
        pieces = chunk_semantic(text, chunk_size)
    else:
        pieces = chunk_fixed(text, chunk_size, overlap=80)

    pieces = [p.strip() for p in pieces if p.strip()]
    if not pieces:
        return 0

    vectors = embed_texts(pieces)
    collection = get_collection()

    ids = [str(uuid.uuid4()) for _ in pieces]
    metadatas = [
        {"filename": filename, "chunk_index": i, "strategy": strategy}
        for i in range(len(pieces))
    ]

    collection.add(ids=ids, embeddings=vectors, documents=pieces, metadatas=metadatas)
    return len(pieces)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vector = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta.get("filename", "unknown"),
            "score": round(1 - distance, 4),  # cosine distance -> similarity
            "metadata": meta,
        })
    return hits
