"""
Intelligent Document QA — single file Streamlit deployment.

Normally I'd split this into separate modules: CSS in a static file, components in their own files, backend logic completely decoupled from
the UI layer. The full version of this project does exactly that, see the src/, api/, and ui/ directories in the GitHub repo.

This file is a self contained version built specifically for Streamlit Cloud deployment, where a single app.py is the entry point and there's
no persistent server process to run the FastAPI backend separately. The tradeoff is intentional: simpler deployment, slightly less clean architecture. 
For production I'd run the FastAPI service and Streamlit dashboard as separate containers, which is how the Docker Compose setup
in the repo works.

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


# page config 

st.set_page_config(
    page_title="Document QA Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# custom css injection 

st.markdown("""
<style>
    /* Global Background and Typography overrides */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    /* Clean, modern styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161920;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #222632;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
        padding: 0px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background-color: #222632;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3345 !important;
        color: #00f2fe !important;
    }
    
    /* Hero header section */
    .hero-container {
        background: linear-gradient(135deg, #161920 0%, #0d0f12 100%);
        border: 1px solid #222632;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .hero-title {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0 !important;
    }
    .hero-subtitle {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }

    /* Premium Modern Cards for Sources & Chat */
    .qa-card {
        background-color: #161920;
        border: 1px solid #222632;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-bubble-container {
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }
    .chat-avatar {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #0d0f12;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        flex-shrink: 0;
        box-shadow: 0 0 10px rgba(0,242,254,0.3);
    }
    .chat-text {
        color: #f1f5f9;
        font-size: 1.05rem;
        line-height: 1.6;
        padding-top: 4px;
    }
    
    /* Passages UI */
    .source-card {
        background-color: #12151c;
        border-left: 4px solid #4facfe;
        border-top: 1px solid #222632;
        border-right: 1px solid #222632;
        border-bottom: 1px solid #222632;
        border-radius: 0 12px 12px 0;
        padding: 18px;
        margin-top: 14px;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        border-bottom: 1px solid #1e2330;
        padding-bottom: 8px;
    }
    .source-title {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .source-score {
        background-color: rgba(79, 142, 247, 0.15);
        color: #38bdf8;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        border: 1px solid rgba(79, 142, 247, 0.3);
    }
    .source-body {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
        font-style: italic;
    }

    /* Metric Layouts */
    .metric-grid {
        display: flex;
        gap: 16px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .metric-box {
        flex: 1;
        min-width: 200px;
        background: #161920;
        border: 1px solid #222632;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-value.cyan { color: #00f2fe; }
    .metric-value.emerald { color: #00f5a0; }
    
    /* Sidebar aesthetic touchups */
    section[data-testid="stSidebar"] {
        background-color: #090b0e;
        border-right: 1px solid #1c202a;
    }
</style>
""", unsafe_allow_html=True)


#constants 

COLLECTION_NAME = "document_store"
CHROMA_DIR = "/tmp/chroma_store"
EVAL_LOG_PATH = "/tmp/eval_log.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using only
the context passages provided. If the context doesn't contain enough information,
say so clearly — don't fill gaps with outside knowledge. Be specific and cite which
passage supports each part of your answer when possible."""


# embedding model (cached so it only loads once)

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# chromadb (cached client)

@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# chunking

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
        sub_tokens = len(enc.encode(sentence))
        if sub_tokens > target_size:
            if current:
                chunks.append(" ".join(current))
                current, current_size = [], 0
            chunks.extend(chunk_text_fixed(sentence, target_size, 50))
            continue
        if current_size + sub_tokens > target_size and current:
            chunks.append(" ".join(current))
            current = [current[-1]] if current else []
            current_size = len(enc.encode(current[0])) if current else 0
        current.append(sentence)
        current_size += sub_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


# document loading

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


# ingestion

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


# retrieval

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


# LLM 

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


# evaluation 

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


# eval log helpers

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


# UI Header Section

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🔍 Intelligent Document QA</h1>
    <p class="hero-subtitle">Upload a document, ask questions, and track how well the answers hold up — built on Claude and ChromaDB.</p>
</div>
""", unsafe_allow_html=True)


# sidebar

with st.sidebar:
    st.header("Configuration")

    # pull from Streamlit secrets first, fall back to manual input
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get yours at console.anthropic.com",
    )

    if api_key:
        st.success("Ready")
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
tab_upload, tab_query, tab_eval = st.tabs(["Upload", "Query", "Evaluation"])


# upload tab

with tab_upload:
    st.subheader("Upload documents")
    st.write("Supports standard PDF and structural plain text files (`.txt`, `.md`). "
             "Uploaded documents will populate the global vector space pipeline.")
    st.info("Tip: Longer documents work better when chunked semantically — keep the default strategy unless you're testing.")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown("<br>", unsafe_allow_html=True)
        ingest_btn = st.button("Ingest", type="primary")
        if ingest_btn:
            total_chunks = 0
            for uploaded_file in uploaded_files:
                with st.spinner(f"Parsing structure for {uploaded_file.name}..."):
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
                        st.success(f"Successfully processed {uploaded_file.name} (Generated {n} chunks)")
                    except Exception as e:
                        st.error(f"{uploaded_file.name} processing exception: {e}")

            if total_chunks > 0:
                st.balloons()
                st.info(f"Ingestion lifecycle completed. Total {total_chunks} chunks stored. You can now use the Query environment.")

    st.divider()
    st.markdown("### Try a sample document")
    st.caption(
        "Don't have a document handy? Load this sample — it covers how RAG systems work, so questions like "What is retrieval precision?" will give meaningful answers."
        " "
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
domain-specific terms, retrieving too few or too many chunks and prompts that don't
anchor the model to the retrieved context."""

    if st.button("Load sample document"):
        with st.spinner("Populating local vector indexes..."):
            n = ingest_document(sample_text, "intro_to_rag.txt", strategy=chunking_strategy)
            st.success(f"Baseline collection updated ({n} chunks). Try running a test prompt: 'What is retrieval precision?'")


# query tab

with tab_query:
    st.subheader("Ask a question")

    if not api_key:
        st.warning("Please configure your Anthropic developer keys inside the sidebar panel to query the system.")
    elif collection.count() == 0:
        st.info("The document index store is currently empty. Please drop or load training files in the Ingestion tab first.")
    else:
        question = st.text_input(
            "Your question",
            placeholder="What would you like to know?",
            label_visibility="collapsed",
        )

        if st.button("Ask", type="primary") and question.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Searching your documents..."):
                try:
                    result = answer_question(question, top_k, api_key)

                    # Custom Chat Response UI Block
                    st.markdown(f"""
                    <div class="qa-card">
                        <div class="chat-bubble-container">
                            <div class="chat-avatar">AI</div>
                            <div class="chat-text">
                                <strong style="color: #00f2fe; font-size: 1.1rem; display:block; margin-bottom: 8px;">Answer</strong>
                                {result["answer"]}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Custom performance matrix display blocks
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value cyan">{result['latency_ms']}ms</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Chunks used</div>
                            <div class="metric-value">{result['chunks_used']}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Model</div>
                            <div class="metric-value">Claude Sonnet</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if result.get("sources"):
                        st.markdown("<br>### Source passages", unsafe_allow_html=True)
                        for i, chunk in enumerate(result["sources"], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">
                                    <span class="source-title">Passage {i} — Source: <strong>{chunk['source']}</strong></span>
                                    <span class="source-score">Relevance score: {chunk['score']:.3f}</span>
                                </div>
                                <div class="source-body">"{chunk['text']}"</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# evaluation tab

with tab_eval:
    st.subheader("Evaluation")
    st.write(
        "Evaluate specific target phrases to assess contextual relevance accuracy. "
        "The diagnostic metrics run real-time checks across retrieval pipelines to log quality deviations."
    )

    if not api_key:
        st.warning("Please configure your Anthropic developer keys inside the sidebar panel to evaluate tracking metrics.")
    elif collection.count() == 0:
        st.info("No documents are available for assessment processing. Populate indexes in the Ingestion tab first.")
    else:
        eval_q = st.text_area(
            "Evaluation question",
            placeholder="Ask something you know the answer to...",
            height=80,
            label_visibility="collapsed",
        )

        if st.button("Run evaluation", type="secondary") and eval_q.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Evaluating... this takes about 10 seconds"):
                try:
                    result = answer_question(eval_q, top_k, api_key)
                    precision_result = score_precision(eval_q, result["sources"], api_key)
                    faith_result = score_faithfulness(
                        eval_q, result["answer"], result["sources"], api_key
                    )

                    # Custom Chat Response UI Block for Evaluation Tab
                    st.markdown(f"""
                    <div class="qa-card">
                        <div class="chat-bubble-container">
                            <div class="chat-avatar">AI</div>
                            <div class="chat-text">
                                <strong style="color: #00f2fe; font-size: 1.1rem; display:block; margin-bottom: 8px;">Answer</strong>
                                {result["answer"]}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    p = precision_result.get("precision", 0) or 0
                    f = faith_result.get("faithfulness", 0) or 0

                    # Modern Styled Metric Cards
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-label">Retrieval Precision</div>
                            <div class="metric-value cyan">{p:.0%}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Answer Faithfulness</div>
                            <div class="metric-value emerald">{f:.0%}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Relevant chunks</div>
                            <div class="metric-value">{precision_result.get('relevant_count', 0)} / {top_k}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value">{result['latency_ms']}ms</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if faith_result.get("explanation"):
                        st.info(f"Faithfulness note: {faith_result['explanation']}")

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
                    st.success("Results saved.")

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

        # Premium Styled Metric Cards for Global History Averages
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Avg precision</div>
                <div class="metric-value cyan">{df['retrieval_precision'].mean():.0%}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg faithfulness</div>
                <div class="metric-value emerald">{df['answer_faithfulness'].mean():.0%}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total runs</div>
                <div class="metric-value">{len(df)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["retrieval_precision"],
            mode="lines+markers", name="Retrieval precision",
            line=dict(color="#00f2fe", width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["answer_faithfulness"],
            mode="lines+markers", name="Answer faithfulness",
            line=dict(color="#00f5a0", width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=True, gridcolor="#1e2330"),
            yaxis=dict(range=[0, 1.05], tickformat=".0%", showgrid=True, gridcolor="#1e2330"),
            height=340,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=1.15, font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        csv = df.to_csv(index=False)
        st.download_button("Export CSV", data=csv,
                           file_name="eval_results.csv", mime="text/csv")

        with st.expander("Raw data"):
            st.dataframe(df[["timestamp", "question", "retrieval_precision",
                              "answer_faithfulness", "latency_ms"]],
                         use_container_width=True)
    else:
        st.info("No recorded assessment steps discovered. Complete standard testing cycles to generate timelines.")
