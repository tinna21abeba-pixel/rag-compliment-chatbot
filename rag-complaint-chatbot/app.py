"""
app.py
------
CrediTrust Financial — Intelligent Complaint Analysis Chatbot
Streamlit interface for the RAG pipeline.

Run with:
    streamlit run app.py

Features:
  - Chat-style Q&A interface
  - Product category filter sidebar
  - Source documents with expandable text
  - Conversation history
  - Loading spinner
  - Error handling
  - Clear conversation button
  - Latency display
"""

from __future__ import annotations

import logging
import os
import sys
import time

import streamlit as st

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import RAGPipeline, DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CrediTrust Complaint Analyst",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean modern look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0d6efd 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 { margin: 0; font-size: 1.8rem; }
    .app-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: #e8f0fe;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
        border-left: 4px solid #0d6efd;
    }
    .assistant-bubble {
        background: #f8f9fa;
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        border-left: 4px solid #198754;
    }

    /* ── Source cards ── */
    .source-card {
        background: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #fd7e14;
    }
    .source-badge {
        display: inline-block;
        background: #0d6efd;
        color: white;
        border-radius: 20px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        margin-right: 0.4rem;
        font-weight: 600;
    }
    .score-badge {
        display: inline-block;
        background: #198754;
        color: white;
        border-radius: 20px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ── Metric cards ── */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        flex: 1;
        text-align: center;
    }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #0d6efd; }
    .metric-card .label { font-size: 0.75rem; color: #6c757d; margin-top: 2px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #f0f2f6; }

    /* ── Input area ── */
    .stTextArea textarea { border-radius: 8px !important; }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []          # list of {"role": ..., "content": ..., "sources": ..., "latency": ...}
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = 0.0


# ---------------------------------------------------------------------------
# Pipeline loader (cached per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(
    retriever_backend: str,
    chroma_persist_dir: str,
    chroma_collection: str,
    generator_backend: str,
    model_id: str,
    top_k: int,
) -> RAGPipeline:
    """Load and cache the RAG pipeline (runs once per Streamlit session)."""
    config = {
        **DEFAULT_CONFIG,
        "retriever_backend": retriever_backend,
        "chroma_persist_dir": chroma_persist_dir,
        "chroma_collection": chroma_collection,
        "generator_backend": generator_backend,
        "model_id": model_id,
        "top_k": top_k,
    }
    return RAGPipeline.from_config(config)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/bank.png",
        width=64,
    )
    st.title("⚙️ Configuration")
    st.markdown("---")

    # — Product filter
    st.subheader("🔍 Product Filter")
    product_options = [
        "All Products",
        "Credit Card",
        "Personal Loan",
        "Savings Account",
        "Money Transfer",
    ]
    selected_product = st.selectbox(
        "Filter complaints by product:",
        product_options,
        help="Restrict the retrieval to a specific product category.",
    )
    product_filter = None if selected_product == "All Products" else selected_product

    # — Retrieval settings
    st.subheader("📊 Retrieval Settings")
    top_k = st.slider(
        "Top-K chunks to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of complaint excerpts retrieved per query.",
    )

    # — Backend settings
    st.subheader("🤖 Model Settings")
    retriever_backend = st.selectbox(
        "Vector Store Backend:",
        ["chromadb", "faiss"],
        help="Must match your persisted vector store format.",
    )
    generator_backend = st.selectbox(
        "LLM Backend:",
        ["hf_hub", "hf_pipeline"],
        help="'hf_hub' uses the HuggingFace Inference API (no GPU needed). "
             "'hf_pipeline' loads the model locally.",
    )
    model_id = st.text_input(
        "Model ID:",
        value="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model identifier.",
    )
    chroma_persist_dir = st.text_input(
        "ChromaDB Path:",
        value="vector_store/chroma",
    )
    chroma_collection = st.text_input(
        "Collection Name:",
        value="cfpb_complaints",
    )

    st.markdown("---")

    # — Load / reload pipeline
    if st.button("🚀 Load Pipeline", type="primary", use_container_width=True):
        with st.spinner("Loading embedder and vector store…"):
            try:
                st.session_state.pipeline = load_pipeline(
                    retriever_backend=retriever_backend,
                    chroma_persist_dir=chroma_persist_dir,
                    chroma_collection=chroma_collection,
                    generator_backend=generator_backend,
                    model_id=model_id,
                    top_k=top_k,
                )
                st.session_state.pipeline_ready = True
                st.success("✅ Pipeline loaded successfully!")
            except Exception as exc:
                st.error(f"❌ Failed to load pipeline:\n\n{exc}")
                st.session_state.pipeline_ready = False

    st.markdown("---")

    # — Session stats
    st.subheader("📈 Session Stats")
    col_a, col_b = st.columns(2)
    col_a.metric("Queries", st.session_state.total_queries)
    col_b.metric(
        "Avg Latency",
        f"{st.session_state.avg_latency:.1f}s"
        if st.session_state.avg_latency > 0
        else "—",
    )

    # — Clear button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.total_queries = 0
        st.session_state.avg_latency = 0.0
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small>CrediTrust Financial · RAG Complaint Analyst · "
        "Powered by Mistral + ChromaDB</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>💳 CrediTrust Complaint Analyst</h1>
        <p>Ask plain-English questions about customer complaints across Credit Cards,
        Personal Loans, Savings Accounts, and Money Transfers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pipeline status banner
if not st.session_state.pipeline_ready:
    st.info(
        "👈 **Configure and load the pipeline** using the sidebar to get started. "
        "Click **Load Pipeline** after adjusting your settings.",
        icon="ℹ️",
    )
else:
    st.success(
        f"✅ Pipeline ready — **{model_id}** via {generator_backend.upper()} · "
        f"Vector store: **{retriever_backend.upper()}** · "
        f"Filter: **{selected_product}**",
        icon="🟢",
    )

# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------
with st.expander("💡 Sample questions to get you started", expanded=False):
    sample_questions = [
        "Why are customers unhappy with credit cards?",
        "What are the most common issues with personal loans?",
        "What problems do customers face when transferring money?",
        "Which companies receive the most complaints about savings accounts?",
        "Are there any recurring billing or payment problems reported?",
        "What fraud-related complaints are customers submitting?",
        "How are customers describing issues with account closures?",
        "What customer service failures are most commonly reported?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state["prefill_question"] = q

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
chat_container = st.container()
with chat_container:
    for turn in st.session_state.conversation:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Sources
            if turn.get("sources"):
                with st.expander(
                    f"📄 View {len(turn['sources'])} retrieved source(s) "
                    f"— latency: {turn.get('latency', 0):.2f}s",
                    expanded=False,
                ):
                    for j, chunk in enumerate(turn["sources"], start=1):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <span class="source-badge">Source {j}</span>
                                <span class="score-badge">Score: {chunk.score:.3f}</span>
                                &nbsp;
                                <strong>{chunk.product_category}</strong> ·
                                {chunk.issue} · {chunk.company} ·
                                ID: <code>{chunk.complaint_id}</code>
                                <br><br>
                                {chunk.text[:500]}{'…' if len(chunk.text) > 500 else ''}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
st.markdown("---")

prefill = st.session_state.pop("prefill_question", "")

with st.container():
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_question = st.text_area(
            "Ask a question about customer complaints:",
            value=prefill,
            placeholder=(
                "e.g. Why are people unhappy with credit card billing? "
                "What fraud issues are reported for money transfers?"
            ),
            height=90,
            key="question_input",
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button(
            "Ask ▶",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.pipeline_ready,
        )

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if submit and user_question and user_question.strip():
    if not st.session_state.pipeline_ready:
        st.error("Please load the pipeline first using the sidebar.")
    else:
        # Add user message to history
        st.session_state.conversation.append(
            {"role": "user", "content": user_question.strip()}
        )

        with st.spinner("🔍 Retrieving relevant complaints and generating answer…"):
            response = st.session_state.pipeline.query(
                question=user_question.strip(),
                top_k=top_k,
                product_filter=product_filter,
            )

        # Update session stats
        st.session_state.total_queries += 1
        prev_avg = st.session_state.avg_latency
        n = st.session_state.total_queries
        st.session_state.avg_latency = (
            (prev_avg * (n - 1) + response.latency_seconds) / n
        )

        # Add assistant message to history
        st.session_state.conversation.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "latency": response.latency_seconds,
            }
        )

        # Rerun to refresh chat display
        st.rerun()

elif submit and not user_question.strip():
    st.warning("Please enter a question before submitting.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top:2rem;">
    <p style="text-align:center; color:#6c757d; font-size:0.8rem;">
        CrediTrust Financial · Internal Tool · Built with Streamlit + LangChain +
        HuggingFace · CFPB Complaint Data
    </p>
    """,
    unsafe_allow_html=True,
)