"""
Timeline Explorer — Ask temporal MCU questions and see events ordered chronologically.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcu_rag.config import MCU_PHASES
from mcu_rag.evaluation.scorer import evaluate
from mcu_rag.generation.pipeline import get_pipeline

st.set_page_config(page_title="Timeline Explorer — MarvelMind", page_icon="📅", layout="wide")

st.title("📅 MCU Timeline Explorer")
st.caption(
    "Ask temporal questions about the MCU — 'What happened before Infinity War?', "
    "'Summarise Phase 2', 'What led to the Snap?' — answers are grounded in the knowledge base."
)

# ── MCU Phase Overview ────────────────────────────────────────────────────────
with st.expander("MCU Phase Reference"):
    phase_cols = st.columns(len(MCU_PHASES))
    for phase_num, films in sorted(MCU_PHASES.items()):
        with phase_cols[phase_num - 1]:
            st.markdown(f"**Phase {phase_num}**")
            for film in films:
                st.caption(f"• {film}")

st.divider()

# ── Query interface ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Timeline Query Examples")
    examples = [
        "What events led to the formation of the Avengers?",
        "Summarise everything that happened in MCU Phase 3",
        "What happened between Infinity War and Endgame?",
        "Explain the MCU timeline from Phase 1 to Phase 4",
        "What is the chronological order of the Infinity Stones appearances?",
        "What happened after The Snap (Decimation)?",
        "Which events in Phase 2 set up Age of Ultron?",
    ]
    st.markdown("Click to use as query:")
    for ex in examples:
        if st.button(ex, key=f"tl_{ex[:25]}"):
            st.session_state.tl_query = ex
            st.rerun()

    st.divider()
    show_trace = st.toggle("Show RAG trace", value=True)
    show_eval  = st.toggle("Show evaluation", value=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "tl_query"   not in st.session_state: st.session_state.tl_query   = ""
if "tl_answer"  not in st.session_state: st.session_state.tl_answer  = ""
if "tl_chunks"  not in st.session_state: st.session_state.tl_chunks  = []
if "tl_trace"   not in st.session_state: st.session_state.tl_trace   = {}
if "tl_rewrite" not in st.session_state: st.session_state.tl_rewrite = ""
if "tl_scores"  not in st.session_state: st.session_state.tl_scores  = {}

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_area(
    "Ask a timeline question",
    value=st.session_state.tl_query,
    placeholder="e.g. What happened in Phase 3?",
    height=80,
)

if st.button("Explore Timeline", type="primary") and query.strip():
    with st.spinner("Retrieving MCU timeline events..."):
        pipeline = get_pipeline()
        result = pipeline.run(
            query=query.strip(),
            mode="timeline",
        )

    st.session_state.tl_query   = query.strip()
    st.session_state.tl_answer  = result["answer"]
    st.session_state.tl_chunks  = result["chunks"]
    st.session_state.tl_trace   = result["trace"]
    st.session_state.tl_rewrite = result["rewritten_query"]

    if result["chunks"]:
        try:
            st.session_state.tl_scores = evaluate(
                query=query.strip(),
                answer=result["answer"],
                chunks=result["chunks"],
                context=result["context"],
            )
        except Exception:
            st.session_state.tl_scores = {}

    st.rerun()

# ── Display answer ─────────────────────────────────────────────────────────────
if st.session_state.tl_answer:
    st.divider()

    # Rewrite trace
    tl_queries = [q.strip() for q in st.session_state.tl_rewrite.split("\n") if q.strip()]
    if tl_queries and tl_queries[0] != st.session_state.tl_query:
        suffix = f" (+{len(tl_queries)-1} variant{'s' if len(tl_queries)-1 != 1 else ''})" if len(tl_queries) > 1 else ""
        st.caption(f"Primary search query: *{tl_queries[0]}*{suffix}")

    # Evaluation scores
    if show_eval and st.session_state.tl_scores:
        s = st.session_state.tl_scores
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunk Relevancy", f"{s.get('chunk_relevancy', 0):.0%}")
        c2.metric("Faithfulness",    f"{s.get('faithfulness', 0):.0%}")
        c3.metric("Chunks Retrieved", len(st.session_state.tl_chunks))

    # Answer
    st.markdown("### Timeline Answer")
    st.markdown(st.session_state.tl_answer)

    # Sources
    chunks = st.session_state.tl_chunks
    if chunks:
        sources = []
        for c in chunks:
            m = c.get("metadata", {})
            s = m.get("title") or m.get("filename") or m.get("source", "")
            if s and s not in sources:
                sources.append(s)
        if sources:
            st.caption("**Sources:** " + " · ".join(f"`{s}`" for s in sources))

    # RAG trace
    if show_trace and st.session_state.tl_trace:
        with st.expander("🔍 RAG Trace"):
            trace = st.session_state.tl_trace

            if st.session_state.tl_rewrite:
                tl_qs = [q.strip() for q in st.session_state.tl_rewrite.split("\n") if q.strip()]
                st.markdown(f"**Primary search query:** `{tl_qs[0]}`")
                for i, v in enumerate(tl_qs[1:], 1):
                    st.markdown(f"**Variant {i}:** `{v}`")

            col_bm25, col_vec = st.columns(2)
            with col_bm25:
                st.markdown("**BM25 top results:**")
                for r in trace.get("bm25_results", [])[:3]:
                    meta = r.get("metadata", {})
                    src = meta.get("title") or meta.get("source", "")
                    st.markdown(f"- `{src}` (score: {r.get('score', 0):.3f})")

            with col_vec:
                st.markdown("**Vector top results:**")
                for r in trace.get("vector_results", [])[:3]:
                    meta = r.get("metadata", {})
                    src = meta.get("title") or meta.get("source", "")
                    st.markdown(f"- `{src}` (score: {r.get('score', 0):.3f})")

            st.markdown(f"**Merged candidates:** {trace.get('merged_count', 0)}")
            st.markdown(f"**After CrossEncoder reranking:** {len(trace.get('reranked', []))}")

            # Reranked chunks
            for i, chunk in enumerate(trace.get("reranked", []), 1):
                meta = chunk.get("metadata", {})
                src  = meta.get("title") or meta.get("source", f"chunk {i}")
                rs   = chunk.get("rerank_score", 0)
                with st.expander(f"Chunk {i}: {src} — rerank={rs:.3f}"):
                    st.markdown(chunk["text"][:400])
else:
    st.info("Enter a timeline question above or click an example in the sidebar.")
