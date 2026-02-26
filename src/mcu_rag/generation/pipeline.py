"""
Full RAG orchestration pipeline.
Ties together: query rewriting → hybrid retrieval → prompt assembly → streaming generation.
"""
from __future__ import annotations

from typing import Any, Generator

import re

from mcu_rag.config import MULTI_QUERY_ENABLED, MULTI_QUERY_VARIANTS, RERANK_TOP_K
from mcu_rag.generation.llm import generate, stream_generate
from mcu_rag.generation.prompts import (
    CHARACTER_PROFILE_TEMPLATE,
    MCU_SYSTEM,
    MULTI_QUERY_EXPANSION_TEMPLATE,
    NO_RAG_TEMPLATE,
    QA_TEMPLATE,
    QUERY_REWRITE_TEMPLATE,
    QUIZ_GEN_TEMPLATE,
    TIMELINE_TEMPLATE,
    format_context,
)
from mcu_rag.retrieval.hybrid import hybrid_retrieve


# ── Step 1: Query rewriting ───────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """
    Use the LLM to rewrite the user query for better document retrieval.
    Enforces preservation of quoted proper nouns (movie titles, character names).
    Falls back to the original query if rewriting fails or drops key entities.
    """
    try:
        prompt = QUERY_REWRITE_TEMPLATE.format(query=query)
        rewritten = generate(prompt, temperature=0.3).strip()
        if len(rewritten) < 5 or len(rewritten) > 500:
            return query
        # If the user quoted a title or name, verify its key words survived
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
        for phrase in quoted:
            words = [w for w in phrase.split() if len(w) > 3]
            if words and not any(w.lower() in rewritten.lower() for w in words):
                return query  # entity dropped — fall back to original
        return rewritten
    except Exception:
        return query


def expand_queries(query: str) -> list[str]:
    """
    Generate alternative query variants for multi-query retrieval.
    Returns up to MULTI_QUERY_VARIANTS strings, or [] on failure.
    """
    if not MULTI_QUERY_ENABLED:
        return []
    try:
        prompt = MULTI_QUERY_EXPANSION_TEMPLATE.format(query=query)
        raw = generate(prompt, temperature=0.5).strip()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        variants = [l for l in lines if 10 <= len(l) <= 400]
        return variants[:MULTI_QUERY_VARIANTS]
    except Exception:
        return []


def _multi_query_retrieve(
    primary: str,
    variants: list[str],
    where_filter: dict | None,
) -> tuple[list[dict], dict]:
    """
    Run hybrid_retrieve for primary + each variant, merge by dedup key.
    Returns merged chunk list (capped at RERANK_TOP_K) and primary trace.
    """
    all_chunks, primary_trace = hybrid_retrieve(primary, where_filter=where_filter)
    seen: dict[str, dict] = {c["text"][:200]: c for c in all_chunks}
    for variant in variants:
        try:
            v_chunks, _ = hybrid_retrieve(variant, where_filter=where_filter)
            for chunk in v_chunks:
                key = chunk["text"][:200]
                if key not in seen:
                    seen[key] = chunk
                elif chunk.get("rerank_score", 0) > seen[key].get("rerank_score", 0):
                    seen[key] = chunk
        except Exception:
            pass  # a failed variant does not poison the result
    merged = sorted(seen.values(), key=lambda c: c.get("rerank_score", 0), reverse=True)
    return merged[:RERANK_TOP_K], primary_trace


# ── Full pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    The main RAG pipeline. Call .run() for a non-streaming answer
    or .stream() for token-by-token streaming.
    """

    def run(
        self,
        query: str,
        mode: str = "qa",           # "qa" | "character" | "timeline" | "no_rag"
        character: str = "",
        topic: str = "",
        num_questions: int = 5,
        metadata_filter: dict | None = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline and return a dict with:
          - answer: str
          - rewritten_query: str
          - chunks: list[dict]  (retrieved chunks)
          - trace: dict          (intermediate retrieval trace)
        """
        # Step 1: Rewrite query
        rewritten = rewrite_query(query) if mode != "no_rag" else query

        # Step 2: Retrieve
        if mode == "no_rag":
            chunks, trace = [], {}
            variants: list[str] = []
        else:
            variants = expand_queries(rewritten)
            if variants:
                chunks, trace = _multi_query_retrieve(rewritten, variants, metadata_filter)
            else:
                chunks, trace = hybrid_retrieve(rewritten, where_filter=metadata_filter)

        # Step 3: Build context + prompt
        context = format_context(chunks) if chunks else ""
        prompt = self._build_prompt(
            query=query,
            context=context,
            mode=mode,
            character=character,
            topic=topic,
            num_questions=num_questions,
        )

        # Step 4: Generate
        system = MCU_SYSTEM if mode != "no_rag" else ""
        answer = generate(prompt, system=system)

        all_queries = [rewritten] + variants
        return {
            "answer":          answer,
            "rewritten_query": "\n".join(all_queries),
            "chunks":          chunks,
            "trace":           trace,
            "context":         context,
        }

    def stream(
        self,
        query: str,
        mode: str = "qa",
        character: str = "",
        topic: str = "",
        metadata_filter: dict | None = None,
    ) -> tuple[Generator[str, None, None], dict[str, Any]]:
        """
        Run the pipeline with streaming generation.
        Returns (token_generator, metadata_dict).
        metadata_dict contains rewritten_query, chunks, and trace — available immediately.
        """
        # Steps 1–3 are synchronous
        rewritten = rewrite_query(query) if mode != "no_rag" else query
        if mode == "no_rag":
            chunks, trace = [], {}
            variants: list[str] = []
        else:
            variants = expand_queries(rewritten)
            if variants:
                chunks, trace = _multi_query_retrieve(rewritten, variants, metadata_filter)
            else:
                chunks, trace = hybrid_retrieve(rewritten, where_filter=metadata_filter)
        context = format_context(chunks) if chunks else ""

        prompt = self._build_prompt(
            query=query,
            context=context,
            mode=mode,
            character=character,
        )
        system = MCU_SYSTEM

        all_queries = [rewritten] + variants
        meta = {
            "rewritten_query": "\n".join(all_queries),
            "chunks":          chunks,
            "trace":           trace,
        }

        # Step 4: Return streaming generator + metadata
        return stream_generate(prompt, system=system), meta

    @staticmethod
    def _build_prompt(
        query: str,
        context: str,
        mode: str,
        character: str = "",
        topic: str = "",
        num_questions: int = 5,
    ) -> str:
        if mode == "character":
            return CHARACTER_PROFILE_TEMPLATE.format(
                character=character or query,
                context=context,
            )
        elif mode == "timeline":
            return TIMELINE_TEMPLATE.format(
                question=query,
                context=context,
            )
        elif mode == "quiz":
            return QUIZ_GEN_TEMPLATE.format(
                context=context,
                topic=topic or query,
                num_questions=num_questions,
            )
        elif mode == "no_rag":
            return NO_RAG_TEMPLATE.format(question=query)
        else:  # default: qa
            return QA_TEMPLATE.format(
                question=query,
                context=context,
            )


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
