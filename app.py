"""Streamlit entry point for the document QA dashboard.

Deployment note: this is set up for Streamlit Cloud, where app.py is the
single process that gets run — there's no separate long-lived server to
host a FastAPI layer alongside it. If this ever needs a real API in front
of it (for a non-Streamlit client, for example), the src/rag_engine
modules are already framework-agnostic and can be wrapped with FastAPI
without touching the dashboard code.
"""

from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.rag_engine.config import CHROMA_DIR
from src.rag_engine.evaluation import (
    load_eval_log,
    save_eval_log,
    score_faithfulness,
    score_precision,
)
from src.rag_engine.loaders import load_pdf, load_text
from src.rag_engine.qa import answer_question
from src.rag_engine.vectorstore import clear_collection, get_collection, ingest_document
from ui.components import render_answer_card, render_metric_grid, render_sources
from ui.styles import DASHBOARD_CSS

SAMPLE_DOC_PATH = Path(__file__).parent / "data" / "sample_docs" / "intro_to_rag.txt"

st.set_page_config(
    page_title="Document QA Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Intelligent Document QA</h1>
    <p class="hero-subtitle">Upload a document, ask questions, and track how well the answers hold up — built on Groq and ChromaDB.</p>
</div>
""", unsafe_allow_html=True)


# --- sidebar: config + collection controls ---------------------------------

with st.sidebar:
    st.header("Configuration")

    api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input(
        "GROQ API Key",
        type="password",
        placeholder="gsk_...",
        help="Get yours at console.groq.com",
    )

    if api_key:
        st.success("Ready")
    else:
        st.warning("Add your API key to get started")

    st.divider()

    collection = get_collection()
    chunk_count = collection.count()
    st.metric("Indexed chunks", chunk_count)

    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        ["sentence-aware", "fixed"],
        help="Sentence-aware avoids cutting normal prose mid-sentence. Fixed splits by token count.",
    )

    st.divider()

    if chunk_count > 0 and st.button("Clear all documents", type="secondary"):
        try:
            clear_collection()
            st.success("Cleared. Refresh the page.")
        except Exception as exc:
            st.error(f"Error: {exc}")


tab_upload, tab_query, tab_eval = st.tabs(["Upload", "Query", "Evaluation"])


# --- upload tab --------------------------------------------------------------

with tab_upload:
    st.subheader("Upload documents")
    st.write(
        "Supports PDF and plain text files (`.txt`, `.md`). Uploaded documents "
        "are split into chunks, embedded, and added to the local vector store."
    )
    st.info("Sentence-aware chunking is a useful default for reports and normal prose.")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Ingest", type="primary"):
        total_chunks = 0
        for uploaded_file in uploaded_files:
            with st.spinner(f"Parsing {uploaded_file.name}..."):
                try:
                    file_bytes = uploaded_file.read()
                    text = load_pdf(file_bytes) if uploaded_file.name.endswith(".pdf") else load_text(file_bytes)
                    n = ingest_document(text=text, filename=uploaded_file.name, strategy=chunking_strategy)
                    total_chunks += n
                    st.success(f"Processed {uploaded_file.name} ({n} chunks)")
                except Exception as exc:
                    st.error(f"{uploaded_file.name} failed: {exc}")

        if total_chunks > 0:
            st.info(f"Done — {total_chunks} chunks stored. Head to the Query tab.")

    st.divider()
    st.markdown("### Try a sample document")
    st.caption("No document handy? Load this one — it covers RAG basics, so a question like "
               "'What is retrieval precision?' will give a meaningful answer.")

    if st.button("Load sample document"):
        with st.spinner("Indexing sample document..."):
            sample_text = SAMPLE_DOC_PATH.read_text()
            n = ingest_document(sample_text, "intro_to_rag.txt", strategy=chunking_strategy)
            st.success(f"Added {n} chunks. Try: 'What is retrieval precision?'")


# --- query tab -----------------------------------------------------------------

with tab_query:
    st.subheader("Ask a question")

    if not api_key:
        st.warning("Add your GROQ API key in the sidebar to ask questions.")
    elif collection.count() == 0:
        st.info("No documents indexed yet. Add some in the Upload tab first.")
    else:
        question = st.text_input(
            "Your question",
            placeholder="What would you like to know?",
            label_visibility="collapsed",
        )

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Searching your documents..."):
                try:
                    result = answer_question(question, top_k, api_key)
                    render_answer_card(result["answer"])
                    render_metric_grid([
                        {"label": "Latency", "value": f"{result['latency_ms']}ms", "accent": "cyan"},
                        {"label": "Chunks used", "value": result["chunks_used"]},
                        {"label": "Model", "value": "Llama 3 (via Groq)"},
                    ])
                    render_sources(result.get("sources", []))
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")


# --- evaluation tab --------------------------------------------------------------

with tab_eval:
    st.subheader("Evaluation")
    st.write(
        "Ask a question you already know the answer to and get retrieval precision "
        "and answer faithfulness scores, judged by a second Llama call."
    )

    if not api_key:
        st.warning("Add your Groq API key in the sidebar to run evaluations.")
    elif collection.count() == 0:
        st.info("No documents indexed yet. Add some in the Upload tab first.")
    else:
        eval_question = st.text_area(
            "Evaluation question",
            placeholder="Ask something you know the answer to...",
            height=80,
            label_visibility="collapsed",
        )

        if st.button("Run evaluation", type="secondary") and eval_question.strip():
            with st.spinner("Evaluating... this takes about 10 seconds"):
                try:
                    result = answer_question(eval_question, top_k, api_key)
                    precision_result = score_precision(eval_question, result["sources"], api_key)
                    faithfulness_result = score_faithfulness(
                        eval_question, result["answer"], result["sources"], api_key
                    )

                    render_answer_card(result["answer"])

                    precision = precision_result.get("precision", 0) or 0
                    faithfulness = faithfulness_result.get("faithfulness", 0) or 0

                    render_metric_grid([
                        {"label": "Retrieval precision", "value": f"{precision:.0%}", "accent": "cyan"},
                        {"label": "Answer faithfulness", "value": f"{faithfulness:.0%}", "accent": "emerald"},
                        {"label": "Relevant chunks", "value": f"{precision_result.get('relevant_count', 0)} / {top_k}"},
                        {"label": "Latency", "value": f"{result['latency_ms']}ms"},
                    ])

                    if faithfulness_result.get("explanation"):
                        st.info(f"Faithfulness note: {faithfulness_result['explanation']}")

                    log = load_eval_log()
                    log.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": eval_question,
                        "retrieval_precision": precision,
                        "answer_faithfulness": faithfulness,
                        "latency_ms": result["latency_ms"],
                        "chunks_used": result["chunks_used"],
                    })
                    save_eval_log(log)
                    st.success("Results saved.")

                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    st.divider()
    st.subheader("Evaluation history")
    log = load_eval_log()

    if not log:
        st.info("No evaluation runs yet. Run one above to start building history.")
    else:
        df = pd.DataFrame(log)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        render_metric_grid([
            {"label": "Avg precision", "value": f"{df['retrieval_precision'].mean():.0%}", "accent": "cyan"},
            {"label": "Avg faithfulness", "value": f"{df['answer_faithfulness'].mean():.0%}", "accent": "emerald"},
            {"label": "Total runs", "value": len(df)},
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["retrieval_precision"],
            mode="lines+markers", name="Retrieval precision",
            line=dict(color="#00f2fe", width=3), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["answer_faithfulness"],
            mode="lines+markers", name="Answer faithfulness",
            line=dict(color="#00f5a0", width=3), marker=dict(size=8),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=True, gridcolor="#1e2330"),
            yaxis=dict(range=[0, 1.05], tickformat=".0%", showgrid=True, gridcolor="#1e2330"),
            height=340,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=1.15, font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

        csv_export = df.to_csv(index=False)
        st.download_button("Export CSV", data=csv_export, file_name="eval_results.csv", mime="text/csv")

        with st.expander("Raw data"):
            st.dataframe(
                df[["timestamp", "question", "retrieval_precision", "answer_faithfulness", "latency_ms"]],
                use_container_width=True,
            )
