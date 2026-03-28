"""
app.py — Phase 5: Streamlit Chat Interface
------------------------------------------
Run locally:  streamlit run app.py
Hosted at:    huggingface.co/spaces/YOUR_USERNAME/ask-my-docs

Features:
  - Chat interface with full history
  - Collapsible source citation cards
  - Pipeline debug panel (expandable)
  - Thumbs up/down feedback logger
  - Document list in sidebar
  - Graceful error handling
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4a90e2;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .score-badge {
        background: #e8f4fd;
        color: #1a6fa8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .feedback-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
FEEDBACK_FILE = "feedback_log.csv"

def log_feedback(question: str, answer: str, rating: str):
    """Append user feedback to a CSV file."""
    file_exists = Path(FEEDBACK_FILE).exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "rating"])
        writer.writerow([
            datetime.now().isoformat(),
            question[:300],
            answer[:500],
            rating,
        ])


def check_ready() -> tuple[bool, str]:
    """Check that all required files and keys exist."""
    missing = []
    if not Path("chroma_db").exists():
        missing.append("chroma_db/ (run: python ingest.py)")
    if not Path("bm25_index.pkl").exists():
        missing.append("bm25_index.pkl (run: python ingest.py)")
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY environment variable")
    if not os.environ.get("COHERE_API_KEY"):
        missing.append("COHERE_API_KEY environment variable")
    if missing:
        return False, "\n".join(f"• {m}" for m in missing)
    return True, ""


def get_doc_list() -> list[str]:
    """List files in docs/ folder."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return []
    return [
        f.name for f in docs_dir.rglob("*")
        if f.suffix.lower() in (".pdf", ".txt", ".md")
    ]


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Ask My Docs")
    st.markdown("---")

    ready, error_msg = check_ready()
    if ready:
        st.success("System ready", icon="✅")
    else:
        st.error("Setup incomplete", icon="⚠️")
        st.markdown(f"**Missing:**\n\n{error_msg}")

    st.markdown("### Indexed Documents")
    doc_list = get_doc_list()
    if doc_list:
        for doc in doc_list:
            st.markdown(f"📎 `{doc}`")
    else:
        st.info("No documents found in docs/")

    st.markdown("---")
    st.markdown("### Pipeline Settings")
    st.caption(f"**Retrieval:** BM25 + ChromaDB (top 20 each)")
    st.caption(f"**Fusion:** Reciprocal Rank Fusion")
    st.caption(f"**Reranker:** Cohere Rerank v3 (top 5)")
    st.caption(f"**LLM:** Groq Llama 3.1 70B")

    st.markdown("---")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    if Path(FEEDBACK_FILE).exists():
        with open(FEEDBACK_FILE, "rb") as f:
            st.download_button(
                "⬇️ Download feedback log",
                f,
                file_name="feedback_log.csv",
                mime="text/csv",
            )


# ── Main area ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Ask My Docs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask any question about your documents. '
    'Every answer includes citations to the exact source.</div>',
    unsafe_allow_html=True,
)

# ── Session state init ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}   # message_index → rating


# ── Render chat history ────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source cards for assistant messages
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander(f"📎 {len(msg['chunks'])} source(s) used", expanded=False):
                for j, chunk in enumerate(msg["chunks"], 1):
                    score_pct = f"{chunk['score']*100:.1f}%"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>Excerpt {j}</strong> — '
                        f'<code>{chunk["source"]}</code>, page {chunk["page"]} '
                        f'<span class="score-badge">relevance {score_pct}</span>'
                        f'<br><br>{chunk["text"][:280]}{"…" if len(chunk["text"])>280 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Show debug steps
            if "steps" in msg:
                with st.expander("🔬 Pipeline debug", expanded=False):
                    steps = msg["steps"]
                    cols = st.columns(4)
                    cols[0].metric("BM25 hits",   steps.get("bm25_hits", "-"))
                    cols[1].metric("Vector hits",  steps.get("vector_hits", "-"))
                    cols[2].metric("After RRF",    steps.get("merged", "-"))
                    cols[3].metric("After rerank", steps.get("reranked", "-"))

            # Feedback buttons
            fb_key = f"fb_{i}"
            existing = st.session_state.feedback.get(i)
            if existing:
                st.caption(f"Feedback recorded: {'👍' if existing=='up' else '👎'} Thank you!")
            else:
                col1, col2, col3 = st.columns([1, 1, 8])
                if col1.button("👍", key=f"up_{i}"):
                    st.session_state.feedback[i] = "up"
                    log_feedback(
                        st.session_state.messages[i-1]["content"],
                        msg["content"],
                        "positive",
                    )
                    st.rerun()
                if col2.button("👎", key=f"dn_{i}"):
                    st.session_state.feedback[i] = "down"
                    log_feedback(
                        st.session_state.messages[i-1]["content"],
                        msg["content"],
                        "negative",
                    )
                    st.rerun()


# ── Chat input ─────────────────────────────────────────────────────────────
if not ready:
    st.warning(
        "Complete the setup steps shown in the sidebar before asking questions.",
        icon="⚠️",
    )
else:
    if prompt := st.chat_input("Ask a question about your documents…"):

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                try:
                    # Import here so module loads after ingestion check
                    from rag_pipeline import query as rag_query
                    result = rag_query(prompt)

                    answer = result["answer"]
                    chunks_data = [
                        {
                            "text":   c.text,
                            "source": c.source,
                            "page":   c.page,
                            "score":  c.score,
                        }
                        for c in result["chunks"]
                    ]
                    steps = result["steps"]

                except FileNotFoundError:
                    answer = (
                        "**Setup required.** The search index was not found. "
                        "Please run `python ingest.py` first."
                    )
                    chunks_data = []
                    steps = {}
                except KeyError as e:
                    answer = (
                        f"**API key missing:** {e}. "
                        "Add your GROQ_API_KEY and COHERE_API_KEY to the .env file "
                        "or HuggingFace Space secrets."
                    )
                    chunks_data = []
                    steps = {}
                except Exception as e:
                    answer = f"**Error:** {str(e)}"
                    chunks_data = []
                    steps = {}

            st.markdown(answer)

            # Source cards
            if chunks_data:
                with st.expander(f"📎 {len(chunks_data)} source(s) used", expanded=False):
                    for j, chunk in enumerate(chunks_data, 1):
                        score_pct = f"{chunk['score']*100:.1f}%"
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>Excerpt {j}</strong> — '
                            f'<code>{chunk["source"]}</code>, page {chunk["page"]} '
                            f'<span class="score-badge">relevance {score_pct}</span>'
                            f'<br><br>{chunk["text"][:280]}{"…" if len(chunk["text"])>280 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Debug panel
            if steps:
                with st.expander("🔬 Pipeline debug", expanded=False):
                    cols = st.columns(4)
                    cols[0].metric("BM25 hits",   steps.get("bm25_hits", "-"))
                    cols[1].metric("Vector hits",  steps.get("vector_hits", "-"))
                    cols[2].metric("After RRF",    steps.get("merged", "-"))
                    cols[3].metric("After rerank", steps.get("reranked", "-"))

        # Save assistant message to history
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "chunks":  chunks_data,
            "steps":   steps,
        })
