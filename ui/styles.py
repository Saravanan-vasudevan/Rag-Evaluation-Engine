"""Custom CSS for the Streamlit dashboard.

Streamlit doesn't support external stylesheets cleanly on Cloud, so this
gets injected via st.markdown(..., unsafe_allow_html=True) once at app
startup. Kept as a single string rather than per-component styling since
Streamlit re-renders the whole script on every interaction.
"""

DASHBOARD_CSS = """
<style>
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }

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

    section[data-testid="stSidebar"] {
        background-color: #090b0e;
        border-right: 1px solid #1c202a;
    }
</style>
"""
