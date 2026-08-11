"""Small HTML snippets shared by the Query and Evaluation tabs.

Both tabs render an answer bubble and a row of metric boxes, and the
original app.py had that markup copy-pasted in both places. Pulling it
out here means the styling only needs to change in one spot.
"""

import streamlit as st


def render_answer_card(answer_text: str) -> None:
    st.markdown(f"""
    <div class="qa-card">
        <div class="chat-bubble-container">
            <div class="chat-avatar">AI</div>
            <div class="chat-text">
                <strong style="color: #00f2fe; font-size: 1.1rem; display:block; margin-bottom: 8px;">Answer</strong>
                {answer_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_grid(metrics: list[dict]) -> None:
    """metrics: [{"label": str, "value": str, "accent": "cyan" | "emerald" | None}]"""
    boxes = ""
    for metric in metrics:
        accent_class = f" {metric['accent']}" if metric.get("accent") else ""
        boxes += f"""
        <div class="metric-box">
            <div class="metric-label">{metric['label']}</div>
            <div class="metric-value{accent_class}">{metric['value']}</div>
        </div>
        """
    st.markdown(f'<div class="metric-grid">{boxes}</div>', unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("### Source passages")
    for i, chunk in enumerate(sources, 1):
        st.markdown(f"""
        <div class="source-card">
            <div class="source-header">
                <span class="source-title">Passage {i} — Source: <strong>{chunk['source']}</strong></span>
                <span class="source-score">Relevance score: {chunk['score']:.3f}</span>
            </div>
            <div class="source-body">"{chunk['text']}"</div>
        </div>
        """, unsafe_allow_html=True)
