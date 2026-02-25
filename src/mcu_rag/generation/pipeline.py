"""
Full RAG orchestration pipeline.
Ties together: query rewriting → hybrid retrieval → prompt assembly → streaming generation.
"""
from __future__ import annotations

from typing import Any, Generator

from mcu_rag.generation.llm import generate, stream_generate
from mcu_rag.generation.prompts import (
    CHARACTER_PROFILE_TEMPLATE,
    MCU_SYSTEM,
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
    Falls back to the original query if rewriting fails.
    """
    try:
        prompt = QUERY_REWRITE_TEMPLATE.format(query=query)
        rewritten = generate(prompt, temperature=0.0).strip()
        # Sanity check: if the model returns something useless, fall back
        if len(rewritten) < 5 or len(rewritten) > 500:
            return query
        return rewritten
    except Exception:
        return query


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
        else:
            chunks, trace = hybrid_retrieve(
                rewritten,
                where_filter=metadata_filter,
            )

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

        return {
            "answer":          answer,
            "rewritten_query": rewritten,
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

        meta = {
            "rewritten_query": rewritten,
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
