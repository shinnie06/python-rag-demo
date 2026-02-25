"""
ChromaDB vector store operations.
Handles initialisation, adding documents, and semantic similarity queries.
"""
from __future__ import annotations

from typing import Any

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from mcu_rag.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    VECTOR_TOP_K,
)


# Sources that emit structured Markdown (headings-aware chunking)
_MARKDOWN_SOURCES = frozenset({"wikipedia", "fandom_wiki"})


def _chunk_documents(documents: list[Document]) -> list:
    """
    Route documents to the appropriate chunker based on their source metadata.

    - wikipedia / fandom_wiki → MarkdownNodeParser (splits at heading boundaries)
      Oversized sections (> CHUNK_SIZE * 4 chars) are sub-split by SentenceSplitter
      so no single node exceeds a useful embedding context size.
    - All other sources → SentenceSplitter (flat sentence-boundary splitting)
    """
    sentence_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    markdown_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=False)
    _max_chars = CHUNK_SIZE * 4  # ~512 tokens; sub-split sections larger than this

    md_docs   = [d for d in documents if d.metadata.get("source") in _MARKDOWN_SOURCES]
    flat_docs = [d for d in documents if d.metadata.get("source") not in _MARKDOWN_SOURCES]

    all_nodes: list = []

    if flat_docs:
        all_nodes.extend(sentence_splitter.get_nodes_from_documents(flat_docs))

    if md_docs:
        raw_nodes = markdown_parser.get_nodes_from_documents(md_docs)
        oversized_docs: list[Document] = []
        for node in raw_nodes:
            if len(node.get_content()) > _max_chars:
                # Re-wrap as Document so SentenceSplitter can sub-split at sentence boundaries
                oversized_docs.append(Document(text=node.get_content(), metadata=node.metadata))
            else:
                all_nodes.append(node)
        if oversized_docs:
            all_nodes.extend(sentence_splitter.get_nodes_from_documents(oversized_docs))

    return all_nodes


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at CHROMA_DIR."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_embed_model() -> OllamaEmbedding:
    """Return the configured Ollama embedding model."""
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_vector_index(documents: list[Document]) -> VectorStoreIndex:
    """
    Chunk documents, embed them, and store in ChromaDB.

    Wikipedia and Fandom wiki documents are chunked with MarkdownNodeParser
    (heading-aware, section-aligned chunks). All other sources use SentenceSplitter.
    Returns a VectorStoreIndex ready for querying.
    """
    chroma_client = get_chroma_client()
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    nodes = _chunk_documents(documents)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


def load_vector_index() -> VectorStoreIndex:
    """
    Load an existing ChromaDB collection as a VectorStoreIndex.
    Raises RuntimeError if the collection is empty (ingestion hasn't run).
    """
    chroma_client = get_chroma_client()

    try:
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)
    except Exception:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_COLLECTION}' not found. "
            "Run: uv run python scripts/ingest.py"
        )

    if chroma_collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Run: uv run python scripts/ingest.py"
        )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def vector_search(
    query: str,
    top_k: int = VECTOR_TOP_K,
    where_filter: dict | None = None,
) -> list[dict]:
    """
    Perform a raw ChromaDB vector search (bypassing LlamaIndex retriever).
    Returns a list of dicts with keys: id, text, metadata, score.
    """
    chroma_client = get_chroma_client()
    embed_model = get_embed_model()

    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return []

    query_embedding = embed_model.get_text_embedding(query)

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    try:
        results = collection.query(**kwargs)
    except Exception:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    hits = []
    ids       = results.get("ids",       [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc_id, text, meta, dist) in enumerate(zip(ids, docs_list, metas, distances)):
        # ChromaDB returns L2 distance; convert to similarity score (0,1]
        # Using 1/(1+d): monotonically decreasing, always positive, no clipping needed
        score = 1.0 / (1.0 + dist)
        hits.append({
            "id":       doc_id,
            "text":     text or "",
            "metadata": meta or {},
            "score":    round(score, 4),
            "rank":     i + 1,
        })

    return hits


def add_documents_to_index(documents: list[Document]) -> int:
    """
    Append new documents to the existing ChromaDB collection without wiping it.
    Used for live PDF ingestion from the Streamlit UI.
    Returns the new total chunk count.
    """
    chroma_client = get_chroma_client()
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )
    return chroma_collection.count()


def get_doc_count() -> int:
    """Return the number of documents in the ChromaDB collection."""
    try:
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
        return collection.count()
    except Exception:
        return 0
