"""
QA Chat page — free-form MCU Q&A with streaming answers, citations, and RAG trace.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcu_rag.evaluation.scorer import evaluate
from mcu_rag.generation.pipeline import get_pipeline

st.set_page_config(page_title="QA Chat — MarvelMind", page_icon="💬", layout="wide")

st.title("💬 MCU Q&A Chat")
st.caption(
    "Ask anything about the Marvel Cinematic Universe. "
    "Every answer is grounded in the knowledge base with source citations."
)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Retrieval Filters")
    st.caption("Optionally narrow retrieval to a specific source or phase.")

    filter_source = st.selectbox(
        "Filter by source",
        ["All sources", "wikipedia", "fandom_wiki", "huggingface_marvel", "pdf_upload"],
    )
    filter_phase = st.selectbox(
        "Filter by MCU Phase",
        ["All phases", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"],
    )

    show_trace = st.toggle("Show RAG trace", value=True)
    show_eval  = st.toggle("Show evaluation scores", value=True)

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Try asking:**")
    examples = [
        "Who is Tony Stark and what are his powers?",
        "What happened during the Battle of New York?",
        "How does Thanos obtain all 6 Infinity Stones?",
        "What is the relationship between Thor and Loki?",
        "Explain the Multiverse Saga in MCU Phase 4.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.pending_query = ex
            st.rerun()


# ── Helper functions (defined BEFORE any calls to them) ──────────────────────

def build_filter() -> dict | None:
    if filter_source != "All sources" and filter_phase != "All phases":
        phase_num = int(filter_phase.split()[-1])
        return {"$and": [{"source": {"$eq": filter_source}}, {"phase": {"$eq": phase_num}}]}
    elif filter_source != "All sources":
        return {"source": {"$eq": filter_source}}
    elif filter_phase != "All phases":
        phase_num = int(filter_phase.split()[-1])
        return {"phase": {"$eq": phase_num}}
    return None


def _render_trace(trace: dict, rewritten_query: str) -> None:
    with st.expander("🔍 RAG Trace — click to inspect the pipeline"):
        if rewritten_query:
            st.markdown(f"**Rewritten query:** `{rewritten_query}`")

        chunks = trace.get("reranked", [])
        if chunks:
            st.markdown(f"**Retrieved {len(chunks)} chunks after reranking:**")
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                source = (
                    meta.get("title") or meta.get("filename")
                    or meta.get("character") or meta.get("source", "?")
                )
                rerank_score = chunk.get("rerank_score", chunk.get("score", 0))
                rrf_score    = chunk.get("rrf_score", 0)
                with st.expander(
                    f"Chunk {i} — {source} | rerank={rerank_score:.3f} | rrf={rrf_score:.5f}"
                ):
                    st.markdown(
                        chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else "")
                    )
                    st.json(meta)


def _render_scores(scores: dict) -> None:
    c1, c2 = st.columns(2)
    relevancy = scores.get("chunk_relevancy", 0)
    faithful  = scores.get("faithfulness", 0)
    c1.metric(
        "Chunk Relevancy", f"{relevancy:.0%}",
        help="Avg cosine similarity: query vs retrieved chunks",
    )
    c2.metric(
        "Faithfulness", f"{faithful:.0%}",
        help="LLM-as-judge: how grounded is the answer?",
    )


# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace") and show_trace:
            _render_trace(msg["trace"], msg.get("rewritten_query", ""))
        if msg.get("scores") and show_eval:
            _render_scores(msg["scores"])

# ── Handle pending query from sidebar examples ────────────────────────────────
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": pending})

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything about the MCU...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        pipeline = get_pipeline()
        metadata_filter = build_filter()

        try:
            token_gen, meta = pipeline.stream(
                query=user_input,
                mode="qa",
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

        # Stream the answer token by token
        full_answer = st.write_stream(token_gen)

        # Show RAG trace
        if show_trace:
            _render_trace(meta.get("trace", {}), meta.get("rewritten_query", ""))

        # Show citations
        chunks = meta.get("chunks", [])
        if chunks:
            sources = []
            for c in chunks:
                m = c.get("metadata", {})
                s = (
                    m.get("title") or m.get("filename")
                    or m.get("character") or m.get("source", "")
                )
                if s and s not in sources:
                    sources.append(s)
            if sources:
                st.caption("**Sources:** " + " · ".join(f"`{s}`" for s in sources))

        # Evaluate response
        scores = {}
        if show_eval and chunks:
            with st.spinner("Evaluating..."):
                try:
                    scores = evaluate(
                        query=user_input,
                        answer=full_answer,
                        chunks=chunks,
                        context="\n".join(c["text"] for c in chunks),
                    )
                    _render_scores(scores)
                except Exception:
                    pass

        # Save message to history
        st.session_state.messages.append({
            "role":            "assistant",
            "content":         full_answer,
            "trace":           meta.get("trace", {}),
            "rewritten_query": meta.get("rewritten_query", ""),
            "scores":          scores,
        })
