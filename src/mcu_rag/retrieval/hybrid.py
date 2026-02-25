"""
Hybrid retrieval: BM25 + Vector → Reciprocal Rank Fusion → CrossEncoder re-ranking.
This is the core of the Advanced RAG pipeline.
"""
from __future__ import annotations

from typing import Any

from mcu_rag.config import (
    BM25_TOP_K,
    CROSS_ENCODER_MODEL,
    RERANK_TOP_K,
    VECTOR_TOP_K,
)
from mcu_rag.retrieval.bm25_retriever import get_bm25_retriever
from mcu_rag.retrieval.vector_store import vector_search


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    Uses text content as the deduplication key.
    k=60 is the standard RRF constant.
    """
    scores: dict[str, float] = {}
    best_hit: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            key = hit["text"][:200]  # use first 200 chars as dedup key
            rrf_score = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in best_hit:
                best_hit[key] = hit

    merged = []
    for key, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
        hit = dict(best_hit[key])
        hit["rrf_score"] = round(rrf_score, 6)
        merged.append(hit)

    return merged


# ── CrossEncoder re-ranker ────────────────────────────────────────────────────

_cross_encoder = None  # lazy-loaded singleton


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        except ImportError as e:
            raise ImportError(
                "Install 'sentence-transformers': uv add sentence-transformers"
            ) from e
    return _cross_encoder


def _rerank(query: str, hits: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Score each hit with the CrossEncoder and return the top_k highest-scoring results.
    Adds a 'rerank_score' key to each hit.
    Falls back to RRF-ordered results if CrossEncoder fails.
    """
    if not hits:
        return []

    try:
        cross_encoder = _get_cross_encoder()
        pairs = [(query, hit["text"]) for hit in hits]
        ce_scores = cross_encoder.predict(pairs)

        for hit, score in zip(hits, ce_scores):
            hit["rerank_score"] = round(float(score), 4)

        reranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return reranked[:top_k]
    except Exception as exc:
        print(f"WARNING: CrossEncoder reranking failed ({exc}); falling back to RRF order.")
        for hit in hits:
            hit.setdefault("rerank_score", 0.0)
        return hits[:top_k]


# ── Public interface ──────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    top_k: int = RERANK_TOP_K,
    where_filter: dict | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Full hybrid retrieval pipeline:
      1. BM25 keyword search  (top BM25_TOP_K)
      2. Vector semantic search (top VECTOR_TOP_K)
      3. Reciprocal Rank Fusion
      4. CrossEncoder re-ranking → top_k final results

    Returns:
        (final_results, trace) where trace contains intermediate results
        for educational display in the RAG Insights page.
    """
    # --- Step 1: BM25 ---
    try:
        bm25_results = get_bm25_retriever().search(query, top_k=BM25_TOP_K)
    except Exception as exc:
        print(f"WARNING: BM25 search failed: {exc}")
        bm25_results = []

    # --- Step 2: Vector search ---
    vector_results = vector_search(query, top_k=VECTOR_TOP_K, where_filter=where_filter)

    # --- Step 3: RRF merge ---
    merged = _reciprocal_rank_fusion(bm25_results, vector_results)

    # --- Step 4: CrossEncoder re-rank ---
    reranked = _rerank(query, merged, top_k=top_k)

    # Build educational trace
    trace = {
        "bm25_results":    bm25_results[:5],
        "vector_results":  vector_results[:5],
        "merged":          merged[:10],   # RRF order before CrossEncoder (for before/after display)
        "merged_count":    len(merged),
        "reranked":        reranked,
    }

    return reranked, trace


def vector_only_retrieve(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Vector-only search — used in RAG Insights comparison panel."""
    return vector_search(query, top_k=top_k)


def bm25_only_retrieve(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """BM25-only search — used in RAG Insights comparison panel."""
    try:
        return get_bm25_retriever().search(query, top_k=top_k)
    except Exception:
        return []
