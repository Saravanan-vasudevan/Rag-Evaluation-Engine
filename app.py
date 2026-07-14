"""
Intelligent Document QA — Hugging Face Spaces deployment.

Single-file version: ingestion, retrieval, evaluation, and UI all in one place.
The FastAPI layer is replaced by direct function calls since we don't need
a separate server process on Spaces.

Drop your documents using the file uploader, ask questions, track eval metrics.
"""

import os
import re
import uuid
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import anthropic
import chromadb
import tiktoken
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "document_store"
CHROMA_DIR = "/tmp/chroma_store"
EVAL_LOG_PATH = "/tmp/eval_log.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using only
the context passages provided. If the context doesn't contain enough information,
say so clearly — don't fill gaps with outside knowledge. Be specific and cite which
passage supports each part of your answer when possible."""


# ── embedding model (cached so it only loads once) ───────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# ── chromadb (cached client) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── chunking ──────────────────────────────────────────────────────────────────

def _get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text_fixed(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    enc = _get_encoder()
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


def chunk_text_semantic(text: str, target_size: int = 400) -> list[str]:
    enc = _get_encoder()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_size = [], [], 0
    for sentence in sentences:
        s_tokens = len(enc.encode(sentence))
        if s_tokens > target_size:
            if current:
                chunks.append(" ".join(current))
                current, current_size = [], 0
            chunks.extend(chunk_text_fixed(sentence, target_size, 50))
            continue
        if current_size + s_tokens > target_size and current:
            chunks.append(" ".join(current))
            current = [current[-1]] if current else []
            current_size = len(enc.encode(current[0])) if current else 0
        current.append(sentence)
        current_size += s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── document loading ──────────────────────────────────────────────────────────

def load_pdf(file_bytes: bytes, filename: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n\n".join(pages)
    finally:
        os.unlink(tmp_path)


def load_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


# ── ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(
    text: str,
    filename: str,
    strategy: str = "semantic",
    chunk_size: int = 400,
) -> int:
    """Chunk, embed, and store a document. Returns number of chunks added."""
    if strategy == "semantic":
        texts = chunk_text_semantic(text, chunk_size)
    else:
        texts = chunk_text_fixed(text, chunk_size, overlap=80)

    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return 0

    embeddings = embed_texts(texts)
    collection = get_chroma_collection()

    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [{"filename": filename, "chunk_index": i, "strategy": strategy}
                 for i in range(len(texts))]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(texts)


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    query_emb = embed_query(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("filename", "unknown"),
            "score": round(1 - dist, 4),
            "metadata": meta,
        })
    return chunks


# ── LLM ──────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def call_claude(user_message: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.2,
    )
    return response.content[0].text


def answer_question(question: str, top_k: int, api_key: str) -> dict:
    start = time.monotonic()
    chunks = retrieve_chunks(question, top_k)

    if not chunks:
        return {
            "answer": "No documents indexed yet. Upload some documents first.",
            "sources": [],
            "latency_ms": 0,
            "chunks_used": 0,
        }

    context = "\n\n---\n\n".join([
        f"[Passage {i+1} — {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"
    answer = call_claude(prompt, api_key)

    return {
        "answer": answer,
        "sources": chunks,
        "latency_ms": int((time.monotonic() - start) * 1000),
        "chunks_used": len(chunks),
    }


# ── evaluation ────────────────────────────────────────────────────────────────

def score_faithfulness(question: str, answer: str, chunks: list[dict], api_key: str) -> dict:
    context = "\n\n".join([c["text"][:400] for c in chunks])
    prompt = f"""Rate whether this answer is faithful to the context (0.0-1.0).
Faithful = only uses info from context. Unfaithful = introduces outside facts.

Context: {context}

Question: {question}
Answer: {answer}

Respond ONLY with JSON: {{"faithfulness": 0.85, "explanation": "brief reason"}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(resp.content[0].text.strip())
    except Exception as e:
        return {"faithfulness": -1.0, "explanation": f"Scoring failed: {e}"}


def score_precision(question: str, chunks: list[dict], api_key: str) -> dict:
    chunk_texts = "\n\n".join([
        f"Chunk {i+1}: {c['text'][:250]}" for i, c in enumerate(chunks)
    ])
    prompt = f"""For each chunk, is it relevant to the question?
Question: {question}
{chunk_texts}

Respond ONLY with JSON array:
[{{"chunk": 1, "relevant": true, "reason": "brief"}}]"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        judgements = json.loads(resp.content[0].text.strip())
        relevant = sum(1 for j in judgements if j.get("relevant", False))
        return {
            "precision": round(relevant / len(chunks), 3),
            "relevant_count": relevant,
            "judgements": judgements,
        }
    except Exception as e:
        return {"precision": -1.0, "relevant_count": 0, "judgements": [], "error": str(e)}


# ── eval log helpers ──────────────────────────────────────────────────────────

def load_eval_log() -> list[dict]:
    if not Path(EVAL_LOG_PATH).exists():
        return []
    try:
        with open(EVAL_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def save_eval_log(records: list[dict]):
    with open(EVAL_LOG_PATH, "w") as f:
        json.dump(records, f)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔍 Document QA Pipeline")
st.caption("Upload documents · Ask questions · Track retrieval quality")

# sidebar
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get yours at console.anthropic.com — free trial credits included",
    )

    if api_key:
        st.success("API key set")
    else:
        st.warning("Add your API key to get started")

    st.divider()

    collection = get_chroma_collection()
    chunk_count = collection.count()
    st.metric("Indexed chunks", chunk_count)

    top_k = st.slider("Chunks to retrieve", 1, 10, 5)

    chunking_strategy = st.selectbox(
        "Chunking strategy",
        ["semantic", "fixed"],
        help="Semantic respects sentence boundaries. Fixed splits by token count.",
    )

    st.divider()

    if chunk_count > 0:
        if st.button("🗑️ Clear all documents", type="secondary"):
            try:
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                client.delete_collection(COLLECTION_NAME)
                get_chroma_collection.cache_clear()
                st.success("Cleared. Refresh the page.")
            except Exception as e:
                st.error(f"Error: {e}")


# tabs
tab_upload, tab_query, tab_eval = st.tabs(["📄 Upload", "💬 Query", "📊 Evaluation"])


# ── upload tab ────────────────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload documents")
    st.write("Supports PDF and plain text (.txt, .md). "
             "Multiple files are fine — they all go into the same index.")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        ingest_btn = st.button("Ingest documents", type="primary")
        if ingest_btn:
            total_chunks = 0
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        if uploaded_file.name.endswith(".pdf"):
                            text = load_pdf(file_bytes, uploaded_file.name)
                        else:
                            text = load_txt(file_bytes)

                        n = ingest_document(
                            text=text,
                            filename=uploaded_file.name,
                            strategy=chunking_strategy,
                        )
                        total_chunks += n
                        st.success(f"{uploaded_file.name} → {n} chunks")
                    except Exception as e:
                        st.error(f"{uploaded_file.name} failed: {e}")

            if total_chunks > 0:
                st.balloons()
                st.info(f"Total: {total_chunks} chunks indexed. Switch to the Query tab.")

    st.divider()
    st.caption(
        "Don't have documents handy? The sample file below covers RAG systems — "
        "good for testing the pipeline end-to-end."
    )

    sample_text = """Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation.
When a user asks a question, the system searches a document store for relevant passages,
then includes those passages in the prompt sent to the language model.

RAG addresses two key problems with language models: knowledge cutoff dates and hallucination.
By retrieving current, specific information at query time, the model can give grounded answers.

Key components: document ingestion (loading and chunking), embedding (converting text to vectors),
vector storage (indexing for similarity search), retrieval (finding relevant chunks),
and generation (the LLM synthesising an answer from retrieved context).

Evaluation is critical. Retrieval precision measures whether retrieved chunks are actually
relevant. Answer faithfulness measures whether the answer stays within the retrieved context.
Tracking these metrics guides improvements to chunking strategy and retrieval parameters.

Chunking strategies: fixed-size splits by token count; semantic splitting respects sentence
boundaries; sliding window uses overlapping chunks to preserve boundary context.

Common failure modes: poor chunking that breaks context, embedding models that miss
domain-specific terms, retrieving too few or too many chunks, and prompts that don't
anchor the model to the retrieved context."""

    if st.button("Load sample document"):
        with st.spinner("Ingesting sample..."):
            n = ingest_document(sample_text, "intro_to_rag.txt", strategy=chunking_strategy)
            st.success(f"Sample ingested → {n} chunks. Try asking: 'What is retrieval precision?'")


# ── query tab ─────────────────────────────────────────────────────────────────

with tab_query:
    st.subheader("Ask a question")

    if not api_key:
        st.warning("Add your Anthropic API key in the sidebar first.")
    elif collection.count() == 0:
        st.info("No documents indexed yet. Go to the Upload tab first.")
    else:
        question = st.text_input(
            "Question",
            placeholder="What is retrieval precision?",
            label_visibility="collapsed",
        )

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Searching and generating..."):
                try:
                    result = answer_question(question, top_k, api_key)

                    st.subheader("Answer")
                    st.write(result["answer"])

                    cols = st.columns(3)
                    cols[0].metric("Latency", f"{result['latency_ms']}ms")
                    cols[1].metric("Chunks used", result["chunks_used"])
                    cols[2].metric("Model", "claude-sonnet")

                    if result.get("sources"):
                        st.divider()
                        st.subheader("Source passages")
                        for i, chunk in enumerate(result["sources"], 1):
                            with st.expander(
                                f"Passage {i} — {chunk['source']} "
                                f"(relevance: {chunk['score']:.3f})"
                            ):
                                st.write(chunk["text"])

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# ── evaluation tab ────────────────────────────────────────────────────────────

with tab_eval:
    st.subheader("Evaluation")
    st.write(
        "Run an evaluation query to score retrieval precision and answer faithfulness. "
        "Results are logged so you can track quality over time."
    )

    if not api_key:
        st.warning("Add your Anthropic API key in the sidebar first.")
    elif collection.count() == 0:
        st.info("No documents indexed. Go to the Upload tab first.")
    else:
        eval_q = st.text_area(
            "Evaluation question",
            placeholder="What are the key components of a RAG system?",
            height=80,
            label_visibility="collapsed",
        )

        if st.button("Run evaluation", type="secondary") and eval_q.strip():
            with st.spinner("Querying and evaluating... (takes ~10 seconds)"):
                try:
                    result = answer_question(eval_q, top_k, api_key)
                    precision_result = score_precision(eval_q, result["sources"], api_key)
                    faith_result = score_faithfulness(
                        eval_q, result["answer"], result["sources"], api_key
                    )

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Scores")
                    cols = st.columns(4)
                    p = precision_result.get("precision", 0) or 0
                    f = faith_result.get("faithfulness", 0) or 0
                    cols[0].metric("Retrieval precision", f"{p:.0%}")
                    cols[1].metric("Answer faithfulness", f"{f:.0%}")
                    cols[2].metric("Relevant chunks",
                                   f"{precision_result.get('relevant_count', 0)}/{top_k}")
                    cols[3].metric("Latency", f"{result['latency_ms']}ms")

                    if faith_result.get("explanation"):
                        st.caption(f"Faithfulness note: {faith_result['explanation']}")

                    # save to log
                    records = load_eval_log()
                    records.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": eval_q,
                        "retrieval_precision": p,
                        "answer_faithfulness": f,
                        "latency_ms": result["latency_ms"],
                        "chunks_used": result["chunks_used"],
                    })
                    save_eval_log(records)
                    st.success("Saved to evaluation log.")

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    # history
    st.divider()
    st.subheader("Evaluation history")
    records = load_eval_log()

    if records:
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        cols = st.columns(3)
        cols[0].metric("Avg precision", f"{df['retrieval_precision'].mean():.0%}")
        cols[1].metric("Avg faithfulness", f"{df['answer_faithfulness'].mean():.0%}")
        cols[2].metric("Total runs", len(df))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["retrieval_precision"],
            mode="lines+markers", name="Retrieval precision",
            line=dict(color="#4F8EF7"),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["answer_faithfulness"],
            mode="lines+markers", name="Answer faithfulness",
            line=dict(color="#56C789"),
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("Export CSV", data=csv,
                           file_name="eval_results.csv", mime="text/csv")

        with st.expander("Raw data"):
            st.dataframe(df[["timestamp", "question", "retrieval_precision",
                              "answer_faithfulness", "latency_ms"]],
                         use_container_width=True)
    else:
        st.info("No evaluation runs yet. Run some queries above.")
