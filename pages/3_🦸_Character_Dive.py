"""
Character Deep-Dive — RAG-powered profile for any MCU character.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcu_rag.config import WIKI_CHARACTERS
from mcu_rag.evaluation.scorer import evaluate
from mcu_rag.generation.pipeline import get_pipeline

st.set_page_config(page_title="Character Dive — MarvelMind", page_icon="🦸", layout="wide")

st.title("🦸 Character Deep-Dive")
st.caption(
    "Select any MCU character and get a comprehensive, source-cited profile "
    "assembled entirely from the knowledge base."
)

# ── Character selection ───────────────────────────────────────────────────────
popular_chars = [
    "Iron Man", "Captain America", "Thor", "Hulk", "Black Widow",
    "Spider-Man", "Doctor Strange", "Black Panther", "Captain Marvel",
    "Thanos", "Loki", "Scarlet Witch", "Guardians of the Galaxy",
    "Ant-Man", "Hawkeye", "Vision", "Star-Lord", "Gamora",
]

col_select, col_custom = st.columns([2, 1])

with col_select:
    character = st.selectbox(
        "Choose a character",
        options=popular_chars,
        index=0,
    )

with col_custom:
    custom_char = st.text_input(
        "Or type a custom character name",
        placeholder="e.g. Nebula, Okoye, Sam Wilson...",
    )
    if custom_char.strip():
        character = custom_char.strip()

# ── Generate button ────────────────────────────────────────────────────────────
if "last_character"  not in st.session_state: st.session_state.last_character  = ""
if "char_profile"    not in st.session_state: st.session_state.char_profile    = ""
if "char_chunks"     not in st.session_state: st.session_state.char_chunks     = []
if "char_rewritten"  not in st.session_state: st.session_state.char_rewritten  = ""
if "char_scores"     not in st.session_state: st.session_state.char_scores     = {}

generate_btn = st.button(f"Generate Profile for {character}", type="primary")

if generate_btn or (character != st.session_state.last_character and st.session_state.char_profile):
    if generate_btn:
        with st.spinner(f"Building RAG profile for **{character}**..."):
            pipeline = get_pipeline()
            result = pipeline.run(
                query=f"Tell me everything about {character} in the MCU",
                mode="character",
                character=character,
            )

        st.session_state.char_profile   = result["answer"]
        st.session_state.char_chunks    = result["chunks"]
        st.session_state.char_rewritten = result["rewritten_query"]
        st.session_state.last_character = character

        if not result["chunks"]:
            st.warning(
                f"No knowledge base content found for **{character}**. "
                "The answer below is generated without RAG context and may not be accurate. "
                "Try running ingestion or uploading a relevant PDF."
            )

        # Evaluate
        if result["chunks"]:
            try:
                scores = evaluate(
                    query=f"Tell me about {character}",
                    answer=result["answer"],
                    chunks=result["chunks"],
                    context=result["context"],
                )
                st.session_state.char_scores = scores
            except Exception:
                st.session_state.char_scores = {}

# ── Render profile ─────────────────────────────────────────────────────────────
if st.session_state.char_profile:
    st.divider()
    st.markdown(f"## {st.session_state.last_character}")

    # Evaluation scores
    scores = st.session_state.char_scores
    if scores:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Chunk Relevancy", f"{scores.get('chunk_relevancy', 0):.0%}")
        sc2.metric("Faithfulness",    f"{scores.get('faithfulness', 0):.0%}")
        sc3.metric("Chunks Retrieved", len(st.session_state.char_chunks))

    st.divider()

    # Profile content
    st.markdown(st.session_state.char_profile)

    st.divider()

    # Sources
    chunks = st.session_state.char_chunks
    if chunks:
        sources = []
        for c in chunks:
            m = c.get("metadata", {})
            s = m.get("title") or m.get("filename") or m.get("character") or m.get("source", "")
            if s and s not in sources:
                sources.append(s)
        if sources:
            st.markdown("**Sources used:**")
            st.caption(" · ".join(f"`{s}`" for s in sources))

        # Raw chunk viewer
        with st.expander(f"View raw retrieved chunks ({len(chunks)} chunks)"):
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                src  = meta.get("title") or meta.get("filename") or meta.get("source", f"chunk {i}")
                rs   = chunk.get("rerank_score", chunk.get("score", 0))
                st.markdown(f"**Chunk {i}** — `{src}` (rerank score: {rs:.3f})")
                st.markdown(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                st.divider()

    # Rewrite trace
    if st.session_state.char_rewritten:
        with st.expander("Query rewriting trace"):
            st.markdown(f"**Original:** Tell me about {st.session_state.last_character}")
            st.markdown(f"**Rewritten:** {st.session_state.char_rewritten}")
else:
    # Placeholder cards for popular characters
    st.divider()
    st.markdown("### Popular Characters")
    cols = st.columns(4)
    for i, char in enumerate(popular_chars[:8]):
        with cols[i % 4]:
            if st.button(char, use_container_width=True, key=f"card_{char}"):
                st.session_state.last_character = char
                with st.spinner(f"Building profile for {char}..."):
                    pipeline = get_pipeline()
                    result = pipeline.run(
                        query=f"Tell me everything about {char} in the MCU",
                        mode="character",
                        character=char,
                    )
                st.session_state.char_profile   = result["answer"]
                st.session_state.char_chunks    = result["chunks"]
                st.session_state.char_rewritten = result["rewritten_query"]
                st.rerun()
