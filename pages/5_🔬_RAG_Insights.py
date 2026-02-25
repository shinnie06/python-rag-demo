"""
RAG Insights — Educational deep-dive: see every step of the pipeline,
compare Vector vs BM25 vs Hybrid, and RAG vs No-RAG side by side.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcu_rag.evaluation.scorer import chunk_relevancy_score, faithfulness_score
from mcu_rag.generation.pipeline import get_pipeline, rewrite_query
from mcu_rag.generation.prompts import format_context
from mcu_rag.retrieval.hybrid import bm25_only_retrieve, hybrid_retrieve, vector_only_retrieve

st.set_page_config(page_title="RAG Insights — MarvelMind", page_icon="🔬", layout="wide")

st.title("🔬 RAG Insights")
st.caption(
    "Educational mode — run any query and inspect every step of the Advanced RAG pipeline. "
    "Compare BM25 vs Vector vs Hybrid search, and see how CrossEncoder reranking changes results."
)

# ── Chunking strategy explainer ────────────────────────────────────────────────
with st.expander("Why chunk boundaries matter — Structure-Aware vs Sentence-Boundary chunking"):
    st.markdown("""
This app uses **two chunking strategies** depending on the data source:

| Source | Chunker | Why |
|--------|---------|-----|
| Wikipedia, Fandom wiki | `MarkdownNodeParser` | Articles have `## Section` headings — split at heading boundaries so each chunk is one coherent topic |
| HuggingFace datasets, PDFs | `SentenceSplitter` (512 tokens) | Flat structured data — sentence-boundary cuts work well |

**Why it matters for retrieval:**
""")
    bad_col, good_col = st.columns(2)
    with bad_col:
        st.error("**SentenceSplitter — arbitrary 512-token cut**")
        st.code(
            "...forged the Iron Man armor in a cave. After escaping,\n"
            "Stark refined the armor and adopted the Iron Man persona.\n"
            "He revealed his identity at a press conference. Tony's\n"
            "closest ally is James Rhodes, a U.S. Air Force officer\n"
            "who becomes War Machine. Pepper Potts serves as his\n"
            "personal assistant and later CEO of Stark Industries...",
            language=None,
        )
        st.caption("Mixes 'Origin story' and 'Relationships' — weak signal for both topics.")
    with good_col:
        st.success("**MarkdownNodeParser — section-aligned chunk**")
        st.code(
            "## Relationships and Allies\n\n"
            "Tony Stark's closest ally is James Rhodes, a U.S. Air\n"
            "Force officer who becomes War Machine. Pepper Potts\n"
            "serves as his personal assistant and later CEO of Stark\n"
            "Industries. Nick Fury recruits Stark into the Avengers\n"
            "Initiative. Stark's AI systems JARVIS and later FRIDAY\n"
            "assist him in managing the Iron Man suits.",
            language=None,
        )
        st.caption("One section, one topic — CrossEncoder scores this much higher for relationship queries.")
    st.info(
        "When you run a query below, look for **section** labels in the retrieved chunks "
        "(e.g. `## Biography`, `## Powers`). Those are heading-aligned chunks from the Markdown pipeline."
    )

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_input(
    "Enter any MCU query to inspect the pipeline",
    placeholder="e.g. How did Thanos collect all Infinity Stones?",
    value="How did Thanos collect all Infinity Stones?",
)

col_run, col_compare = st.columns([1, 2])
with col_run:
    run_btn = st.button("Run Full Pipeline", type="primary", use_container_width=True)
with col_compare:
    compare_rag = st.toggle("RAG vs No-RAG comparison", value=False)

st.divider()

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and query.strip():
    q = query.strip()

    # ── Step 1: Query Rewriting ───────────────────────────────────────────
    st.markdown("## Step 1 — Query Rewriting")
    st.caption(
        "Before searching, the LLM rewrites your question to use precise MCU terminology "
        "and remove ambiguity. This improves what gets retrieved."
    )

    with st.expander("How does query rewriting work? (show prompt)"):
        st.markdown("**Why rewrite?** Conversational queries are vague. Retrieval works best with specific, dense terminology.")
        st.markdown("**Prompt sent to the LLM:**")
        st.code(
            "You are a query optimisation assistant for a Marvel MCU knowledge base.\n"
            "Rewrite the following user query to improve document retrieval accuracy.\n"
            "Make the rewritten query more specific, include relevant MCU terminology,\n"
            "and remove ambiguity.\n"
            "Output ONLY the rewritten query — no explanation, no quotes.\n\n"
            "Original query: {your query}\n"
            "Rewritten query:",
            language=None,
        )
        st.markdown("""
**Examples of what changes:**

| Original query | Rewritten for retrieval |
|---|---|
| "What did the bad guy do?" | "What were Thanos's actions and motivations in Avengers: Infinity War?" |
| "Who's the spider guy?" | "Who is Spider-Man / Peter Parker in the MCU and what are his powers?" |
| "RDJ coming back?" | "Is Robert Downey Jr. returning to the MCU and in what role?" |
""")

    col_orig, col_arrow, col_rewrite = st.columns([2, 0.3, 2])
    with col_orig:
        st.markdown("**Your original query**")
        st.info(q)
    with col_arrow:
        st.markdown("<div style='text-align:center; font-size:2em; padding-top:1.5em'>→</div>", unsafe_allow_html=True)
    with col_rewrite:
        with st.spinner("Rewriting query..."):
            rewritten = rewrite_query(q)
        st.markdown("**Rewritten for retrieval**")
        st.success(rewritten)
        if rewritten == q:
            st.caption("No change — query was already retrieval-ready.")
        else:
            st.caption("This rewritten version will retrieve more targeted chunks from the knowledge base.")

    st.divider()

    # ── Step 2: Retrieval comparison ──────────────────────────────────────
    st.markdown("## Step 2 — Retrieval Comparison")
    st.caption("Three methods run in parallel on the rewritten query. The Hybrid column is what actually goes to the LLM.")

    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.info(
            "**🔍 BM25 — Keyword Match**\n\n"
            "Scores chunks by exact word frequency (like a search engine). "
            "Great for character names and specific terms. "
            "Misses synonyms — 'snap' won't match 'decimation'."
        )
    with exp2:
        st.info(
            "**🧠 Vector — Semantic Match**\n\n"
            "Converts your query to a number vector using `nomic-embed-text` "
            "and finds chunks that *mean* the same thing, even with different words. "
            "Handles paraphrasing but can make meaning-based false positives."
        )
    with exp3:
        st.success(
            "**⚡ Hybrid — Final Selection**\n\n"
            "Merges BM25 + Vector via **RRF** (chunks in both lists rank higher), "
            "then **CrossEncoder** reads each candidate with your query to give a precise "
            "relevance score. This is what the LLM actually sees."
        )

    with st.spinner("Running BM25, Vector, and Hybrid retrieval..."):
        bm25_results    = bm25_only_retrieve(rewritten, top_k=5)
        vector_results  = vector_only_retrieve(rewritten, top_k=5)
        hybrid_results, trace = hybrid_retrieve(rewritten)

    # Overlap indicator
    bm25_keys   = {r["text"][:120] for r in bm25_results}
    vector_keys = {r["text"][:120] for r in vector_results}
    overlap_keys = bm25_keys & vector_keys
    if overlap_keys:
        st.success(
            f"{len(overlap_keys)} chunk(s) appeared in **both** BM25 and Vector results "
            "— these get the strongest boost in Hybrid ranking."
        )
    else:
        st.info("BM25 and Vector found different chunks — each method contributed unique results to Hybrid.")

    def _render_result_list(results: list[dict], shared_keys: set | None = None) -> None:
        if not results:
            st.caption("No results returned.")
            return
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            src  = meta.get("title") or meta.get("filename") or meta.get("source", f"chunk {i}")
            sc   = r.get("score", r.get("raw_score", 0))
            section_label = meta.get("section_summary") or meta.get("header_path") or ""
            section_tag = f" › `{section_label}`" if section_label else ""
            is_shared = bool(shared_keys and r["text"][:120] in shared_keys)
            overlap_tag = " 🔗" if is_shared else ""
            with st.expander(f"#{i}{overlap_tag}  {src}{section_tag}  (score: {sc:.3f})"):
                if is_shared:
                    st.caption("🔗 Also retrieved by the other method — boosted in Hybrid ranking.")
                if section_label:
                    st.caption(f"Section: {section_label}")
                st.markdown(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("🔍 **BM25 — Keyword Match**")
        _render_result_list(bm25_results, shared_keys=overlap_keys)
    with rc2:
        st.markdown("🧠 **Vector — Semantic Match**")
        _render_result_list(vector_results, shared_keys=overlap_keys)
    with rc3:
        st.markdown("⚡ **Hybrid + CrossEncoder — Final**")
        _render_result_list(hybrid_results, shared_keys=None)

    with st.expander("Why do some results look irrelevant?"):
        st.markdown("""
**BM25** matches keywords regardless of context. A query about the "Power Stone" might pull a chunk
about Tony Stark's "power armour" because the word *power* appears frequently in both. BM25 has no
understanding of meaning — only word counts.

**Vector search** can also occasionally retrieve semantically similar but topically wrong chunks —
e.g. two different battles described with similar action language. The embedder doesn't always
distinguish between "Thanos snapping" and "Tony snapping his fingers."

**This is exactly why CrossEncoder reranking exists** (Step 3). It reads your query and each chunk
together and scores them more precisely — effectively catching the false positives that BM25 and
Vector let through. The Hybrid + CrossEncoder column is the cleaned-up final result.
""")

    st.divider()

    # ── Step 3: CrossEncoder Reranking ────────────────────────────────────
    st.markdown("## Step 3 — CrossEncoder Reranking")
    st.caption(
        f"CrossEncoder scored all {trace.get('merged_count', '?')} merged candidates "
        f"and selected the top {len(hybrid_results)} chunks for the LLM."
    )

    with st.expander("How to read this chart + what is CrossEncoder?"):
        st.markdown("""
**Standard embedding (bi-encoder)** encodes the query and each chunk *separately*, then compares
the vectors. Fast enough to scan thousands of chunks — but loses nuance because the model never
sees them together.

**CrossEncoder** reads your query AND the chunk *in a single pass*, like a human reading both at
once. It outputs one precise relevance score per chunk. Much more accurate, but too slow to run
on the full database — so we only apply it to the top candidates after Hybrid retrieval.

**How to read the chart:**

| Element | What it means |
|---|---|
| 🔴 **Red bar** | Chunk ranked #1 — the most relevant, sent first to the LLM |
| 🔵 **Blue bars** | Other selected chunks — all sent to LLM as context |
| **Source** (y-axis) | Which Wikipedia / Fandom article or dataset the chunk came from |
| **Score** (x-axis) | CrossEncoder relevance score — **only relative order matters**, not the absolute number. Scores can be negative. |

**What to look for:** chunks that were ranked low by BM25/Vector but jumped to the top after
CrossEncoder — that's the reranker correcting false positives.
""")

    reranked = trace.get("reranked", hybrid_results)
    merged_before = trace.get("merged", [])

    if reranked:
        # ── Bar chart ──────────────────────────────────────────────────────
        try:
            import plotly.graph_objects as go  # type: ignore

            sources, scores = [], []
            for r in reranked:
                meta = r.get("metadata", {})
                sources.append((meta.get("title") or meta.get("source", "?"))[:40])
                scores.append(r.get("rerank_score", r.get("score", 0)))

            fig = go.Figure(go.Bar(
                x=scores,
                y=sources,
                orientation="h",
                marker_color=["#e23636"] + ["#6b8cce"] * (len(scores) - 1),
                text=[f"{s:.2f}" for s in scores],
                textposition="outside",
            ))
            fig.update_layout(
                title="CrossEncoder Scores — Higher = More Relevant to Your Query",
                xaxis_title="CrossEncoder score (higher = more relevant; can be negative — only order matters)",
                yaxis_title="Source article",
                height=max(40 * len(sources) + 160, 220),
                margin=dict(l=10, r=100, t=50, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
                xaxis=dict(zeroline=True, zerolinecolor="#666"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "🔴 Red = #1 chunk (most relevant, LLM reads this first)  |  "
                "🔵 Blue = other context chunks  |  "
                "Source = article the chunk came from  |  "
                "Score = CrossEncoder rating (only relative order matters)"
            )
        except ImportError:
            for i, r in enumerate(reranked, 1):
                meta = r.get("metadata", {})
                src = meta.get("title") or meta.get("source", "?")
                sc  = r.get("rerank_score", r.get("score", 0))
                icon = "🔴" if i == 1 else "🔵"
                st.markdown(f"{icon} #{i} **{src}** — score: `{sc:.2f}`")

        # ── Before vs After reranking ───────────────────────────────────────
        if merged_before:
            st.markdown("#### Before vs After CrossEncoder")
            st.caption(
                "The left column shows the RRF-merged order before CrossEncoder. "
                "The right column is the final order after. Watch for position changes — "
                "that's the reranker doing its job."
            )
            before_col, after_col = st.columns(2)

            # Build a map: text_key → final rank
            final_rank_map = {r["text"][:120]: i for i, r in enumerate(reranked, 1)}

            with before_col:
                st.markdown("**Before CrossEncoder** *(RRF order)*")
                for i, r in enumerate(merged_before, 1):
                    meta = r.get("metadata", {})
                    src  = (meta.get("title") or meta.get("source", "?"))[:32]
                    rrf  = r.get("rrf_score", 0)
                    key  = r["text"][:120]
                    after_pos = final_rank_map.get(key)
                    if after_pos is not None:
                        if after_pos < i:
                            arrow = f"🔼 → Final **#{after_pos}**"
                        elif after_pos > i:
                            arrow = f"🔽 → Final **#{after_pos}**"
                        else:
                            arrow = f"➡️ → Final **#{after_pos}**"
                        st.markdown(f"`#{i}` {src}  `rrf={rrf:.5f}` {arrow}")
                    else:
                        st.markdown(f"`#{i}` {src}  `rrf={rrf:.5f}` — dropped by reranker")

            with after_col:
                st.markdown("**After CrossEncoder** *(final selection)*")
                for i, r in enumerate(reranked, 1):
                    meta  = r.get("metadata", {})
                    src   = (meta.get("title") or meta.get("source", "?"))[:32]
                    score = r.get("rerank_score", 0)
                    icon  = "🔴" if i == 1 else "🔵"
                    st.markdown(f"{icon} `#{i}` {src}  `ce={score:.2f}`")

    st.divider()

    # ── Step 4: Merged context ────────────────────────────────────────────
    st.markdown("## Step 4 — Context Assembled for LLM")
    context = format_context(hybrid_results)
    with st.expander("View full context sent to LLM"):
        st.text(context)

    st.divider()

    # ── Step 5: Generation ────────────────────────────────────────────────
    st.markdown("## Step 5 — LLM Generation")

    if compare_rag:
        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            st.markdown("**With RAG (grounded)**")
            pipeline = get_pipeline()
            with st.spinner("Generating RAG answer..."):
                rag_result = pipeline.run(query=q, mode="qa")
            st.markdown(rag_result["answer"])

        with gen_col2:
            st.markdown("**Without RAG (LLM only)**")
            with st.spinner("Generating no-RAG answer..."):
                no_rag_result = pipeline.run(query=q, mode="no_rag")
            st.markdown(no_rag_result["answer"])

        st.divider()
        st.markdown("### Comparison: RAG vs No-RAG")
        st.caption(
            "Notice: the RAG answer includes specific citations and sourced details. "
            "The No-RAG answer relies on the LLM's training data and may hallucinate or be vague."
        )
    else:
        pipeline = get_pipeline()
        with st.spinner("Generating answer..."):
            rag_result = pipeline.run(query=q, mode="qa")
        st.markdown(rag_result["answer"])

    st.divider()

    # ── Step 6: Evaluation ─────────────────────────────────────────────────
    st.markdown("## Step 6 — Evaluation Metrics")

    with st.spinner("Computing evaluation scores..."):
        relevancy = chunk_relevancy_score(q, hybrid_results)
        answer_text = rag_result["answer"]
        faithful  = faithfulness_score(answer_text, context)

    ev1, ev2, ev3 = st.columns(3)
    ev1.metric(
        "Chunk Relevancy",
        f"{relevancy:.0%}",
        help="Average cosine similarity between query embedding and retrieved chunk embeddings. Higher = more relevant chunks retrieved.",
    )
    ev2.metric(
        "Faithfulness",
        f"{faithful:.0%}",
        help="LLM-as-judge: how well is the answer grounded in the provided context? Higher = less hallucination.",
    )
    ev3.metric(
        "Chunks Used",
        len(hybrid_results),
        help="Number of chunks passed to the LLM as context.",
    )

    if relevancy >= 0.7 and faithful >= 0.7:
        st.success("High-quality RAG response — grounded and relevant!")
    elif relevancy >= 0.5 or faithful >= 0.5:
        st.info("Moderate quality — the retrieval found relevant content.")
    else:
        st.warning("Low scores — the knowledge base may not have sufficient coverage for this query.")

    st.divider()

    # ── Pipeline Summary ───────────────────────────────────────────────────
    st.markdown("## Pipeline Summary")
    summary_data = {
        "Step":   ["1. Query Rewriting", "2. BM25 Results", "3. Vector Results", "4. After RRF Merge", "5. After CrossEncoder", "6. Chunk Relevancy", "7. Faithfulness"],
        "Result": [
            rewritten,
            f"{len(bm25_results)} chunks",
            f"{len(vector_results)} chunks",
            f"{trace.get('merged_count', '?')} candidates",
            f"{len(hybrid_results)} final chunks",
            f"{relevancy:.0%}",
            f"{faithful:.0%}",
        ],
    }
    try:
        import pandas as pd  # type: ignore
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
    except Exception:
        for step, result in zip(summary_data["Step"], summary_data["Result"]):
            st.markdown(f"- **{step}**: {result}")

else:
    st.info("Enter a query above and click 'Run Full Pipeline' to see the complete Advanced RAG pipeline in action.")
    st.markdown("""
    ### What this page shows you:

    | Step | What you'll see |
    |------|----------------|
    | **Chunking strategy** | Why Wikipedia/Fandom chunks are section-aligned vs sentence-split (see expander above) |
    | **1. Query Rewriting** | How the LLM transforms your question for better retrieval |
    | **2. Retrieval Comparison** | BM25 (keyword) vs Vector (semantic) results side by side |
    | **3. CrossEncoder Reranking** | Bar chart showing how neural re-ranking changes the ranking |
    | **4. Context Assembly** | The exact text chunks sent to the LLM |
    | **5. Generation** | The grounded answer — optionally vs a No-RAG baseline |
    | **6. Evaluation** | Chunk relevancy + faithfulness scores with interpretation |

    This is the power of **Advanced RAG** vs naive RAG — each step improves answer quality.
    """)
