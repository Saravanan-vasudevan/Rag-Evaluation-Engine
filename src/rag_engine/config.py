"""Shared constants for the RAG pipeline.

Pulled out of app.py so the ingestion, retrieval and eval modules don't
each hardcode their own copy of the collection name / model IDs.
"""

import os

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/tmp/chroma_store")
EVAL_LOG_PATH = os.environ.get("EVAL_LOG_PATH", "/tmp/eval_log.json")
COLLECTION_NAME = "document_store"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ANSWER_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "llama-3.1-8b-instant"  # cheaper model for eval scoring, doesn't need to be as strong

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using only
the context passages provided. If the context doesn't contain enough information,
say so clearly — don't fill gaps with outside knowledge. Be specific and cite which
passage supports each part of your answer when possible."""
