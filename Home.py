"""
MarvelMind — MCU RAG Showcase
Home page: app status, pipeline overview, PDF upload.
"""
import sys
from pathlib import Path

import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mcu_rag.generation.llm import is_ollama_running, list_available_models
from mcu_rag.retrieval.vector_store import get_doc_count
from mcu_rag.config import LLM_MODEL, EMBED_MODEL, UPLOADS_DIR

st.set_page_config(
    page_title="MarvelMind — MCU RAG",
    page_icon="🦸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🦸 MarvelMind")
st.subheader("Advanced RAG Showcase powered by the Marvel Cinematic Universe")
st.markdown(
    "A fully **local**, privacy-preserving RAG system — no API keys, no cloud, "
    "no internet required after setup. Built with **Ollama + ChromaDB + LlamaIndex + Streamlit**."
)

st.divider()

# ── Status Dashboard ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Ollama LLM")
    llm_ok = is_ollama_running(LLM_MODEL)
    if llm_ok:
        st.success(f"Running — `{LLM_MODEL}`")
    else:
        st.error(f"Not found — `{LLM_MODEL}`")
        st.caption(f"Run: `ollama pull {LLM_MODEL}`")

    available = list_available_models()
    if available:
        with st.expander("Available models"):
            for m in available:
                st.code(m, language=None)

with col2:
    st.markdown("### Embeddings")
    embed_ok = is_ollama_running(EMBED_MODEL)
    if embed_ok:
        st.success(f"Running — `{EMBED_MODEL}`")
    else:
        st.error(f"Not found — `{EMBED_MODEL}`")
        st.caption(f"Run: `ollama pull {EMBED_MODEL}`")

with col3:
    st.markdown("### Knowledge Base")
    doc_count = get_doc_count()
    if doc_count > 0:
        st.success(f"{doc_count:,} chunks indexed")
    else:
        st.error("Empty — run ingestion first")
        st.caption("Run: `uv run python scripts/ingest.py`")

st.divider()

# ── RAG Pipeline Diagram ──────────────────────────────────────────────────────
st.markdown("## How MarvelMind Works")
st.markdown("This app demonstrates **Advanced RAG** — every step is visible in the RAG Insights page.")

pipeline_cols = st.columns(7)
steps = [
    ("1", "Query\nRewriting", "LLM rewrites your query for better retrieval"),
    ("2", "BM25\nSearch", "Keyword-based search over all chunks"),
    ("3", "Vector\nSearch", "Semantic similarity search via embeddings"),
    ("4", "RRF\nFusion", "Reciprocal Rank Fusion merges both result lists"),
    ("5", "CrossEncoder\nReranking", "Neural re-ranker selects the best 5 chunks"),
    ("6", "LLM\nGeneration", "Ollama generates a grounded, cited answer"),
    ("7", "Evaluation", "Faithfulness & relevancy scores displayed"),
]

for col, (num, title, desc) in zip(pipeline_cols, steps):
    with col:
        st.markdown(
            f"""
            <div style="
                background: #1a1a2e;
                border: 1px solid #e23636;
                border-radius: 10px;
                padding: 12px 8px;
                text-align: center;
                min-height: 110px;
            ">
                <div style="font-size: 1.4em; font-weight: bold; color: #e23636;">{num}</div>
                <div style="font-size: 0.8em; font-weight: 600; color: #fff; margin: 4px 0;">{title}</div>
                <div style="font-size: 0.65em; color: #aaa;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Navigation Guide ──────────────────────────────────────────────────────────
st.markdown("## Explore the App")

nav_cols = st.columns(2)
with nav_cols[0]:
    st.markdown("""
    | Page | What you can do |
    |------|----------------|
    | 💬 QA Chat | Ask anything about the MCU with streaming answers and source citations |
    | 🧠 Quiz Mode | Test your MCU knowledge with AI-generated trivia from the knowledge base |
    | 🦸 Character Dive | Deep-dive profile for any MCU character — sourced from the knowledge base |
    | 📅 Timeline Explorer | Ask temporal questions like "What happened after Infinity War?" |
    | 🔬 RAG Insights | Educational mode — see every step of the pipeline live |
    """)

with nav_cols[1]:
    st.markdown("### Tech Stack")
    st.markdown("""
    - **LLM**: `llama3.1:8b` via [Ollama](https://ollama.com) (local, free)
    - **Embeddings**: `nomic-embed-text` (8K context, fast)
    - **Vector Store**: ChromaDB (persistent, local)
    - **RAG Framework**: LlamaIndex
    - **Hybrid Search**: BM25 + Vector + CrossEncoder
    - **Evaluation**: LLM-as-judge + cosine similarity
    - **UI**: Streamlit
    - **Package manager**: uv
    """)

st.divider()

# ── PDF Upload ────────────────────────────────────────────────────────────────
st.markdown("## Add to the Knowledge Base")
st.markdown(
    "Upload MCU-related PDFs (character guides, scripts, wikis) and they will be ingested live."
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Files are saved to data/uploads/ and ingested into the knowledge base immediately.",
)

if uploaded_files:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    newly_added = []

    for uf in uploaded_files:
        target = UPLOADS_DIR / uf.name
        if not target.exists():
            target.write_bytes(uf.read())
            newly_added.append(uf.name)
        else:
            st.caption(f"Already uploaded: {uf.name}")

    if newly_added:
        st.success(f"Saved: {', '.join(newly_added)}")
        with st.spinner("Ingesting PDFs into knowledge base..."):
            try:
                from mcu_rag.ingestion.pdf_loader import ingest_single_pdf
                from mcu_rag.retrieval.vector_store import add_documents_to_index, get_chroma_client
                from mcu_rag.retrieval.bm25_retriever import BM25Retriever
                from mcu_rag.config import CHROMA_COLLECTION

                all_docs = []
                for fname in newly_added:
                    docs = ingest_single_pdf(UPLOADS_DIR / fname)
                    all_docs.extend(docs)

                if all_docs:
                    new_count = add_documents_to_index(all_docs)

                    # Rebuild BM25 index so it stays in sync with ChromaDB
                    client = get_chroma_client()
                    collection = client.get_collection(CHROMA_COLLECTION)
                    total = collection.count()
                    result = collection.get(limit=total, include=["documents", "metadatas"])
                    chunk_dicts = [
                        {"text": t, "metadata": m or {}}
                        for t, m in zip(result.get("documents", []), result.get("metadatas", []))
                        if t
                    ]
                    bm25 = BM25Retriever()
                    bm25.build(chunk_dicts)
                    bm25.save()

                    st.success(
                        f"Ingested {len(all_docs)} pages. "
                        f"Knowledge base now has {new_count:,} chunks. "
                        "BM25 index updated. Ready to query!"
                    )
                    st.rerun()
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "MarvelMind — Built for an AI course mini project showcase | "
    "Powered by Ollama + ChromaDB + LlamaIndex + Streamlit | "
    "All Marvel content referenced for educational purposes."
)
