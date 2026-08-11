"""Sentence-transformer wrapper.

Streamlit's cache_resource decorator lives here instead of app.py so the
model only gets loaded once per session regardless of which tab triggers it.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
