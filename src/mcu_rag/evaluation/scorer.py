"""
Local evaluation metrics — no external API required.

Metrics:
  - chunk_relevancy:  average cosine similarity between query embedding and retrieved chunk embeddings
  - faithfulness:     LLM-as-judge score (0–1) measuring how grounded the answer is in the context
"""
from __future__ import annotations

import numpy as np

from mcu_rag.config import EMBED_MODEL, OLLAMA_BASE_URL
from mcu_rag.generation.llm import generate
from mcu_rag.generation.prompts import FAITHFULNESS_JUDGE_TEMPLATE


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the Ollama embedding model."""
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        return [embed_model.get_text_embedding(t) for t in texts]
    except Exception:
        return [[] for _ in texts]


def chunk_relevancy_score(query: str, chunks: list[dict]) -> float:
    """
    Average cosine similarity between the query embedding and each chunk embedding.
    Returns a float in [0, 1].
    """
    if not chunks:
        return 0.0

    texts = [c["text"] for c in chunks if c.get("text")]
    if not texts:
        return 0.0

    try:
        all_texts = [query] + texts
        embeddings = _embed(all_texts)
        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        if not query_emb:
            return 0.0

        sims = [_cosine_similarity(query_emb, ce) for ce in chunk_embs if ce]
        return round(float(np.mean(sims)), 4) if sims else 0.0
    except Exception:
        return 0.0


def faithfulness_score(answer: str, context: str) -> float:
    """
    LLM-as-judge faithfulness score.
    Asks the LLM to rate (1–5) how grounded the answer is in the context.
    Normalised to [0, 1].
    Returns 0.0 if evaluation fails.
    """
    if not answer.strip() or not context.strip():
        return 0.0

    try:
        prompt = FAITHFULNESS_JUDGE_TEMPLATE.format(
            context=context[:3000],   # cap to avoid token overflow
            answer=answer[:1000],
        )
        raw = generate(prompt, temperature=0.0).strip()

        # Extract first integer from the response
        import re
        match = re.search(r"\b([1-5])\b", raw)
        if match:
            score_int = int(match.group(1))
            return round((score_int - 1) / 4.0, 4)  # normalise 1-5 → 0-1
        return 0.0
    except Exception:
        return 0.0


def evaluate(
    query: str,
    answer: str,
    chunks: list[dict],
    context: str,
) -> dict[str, float]:
    """
    Run both metrics and return a dict:
      {"chunk_relevancy": float, "faithfulness": float}
    """
    relevancy = chunk_relevancy_score(query, chunks)
    faithful  = faithfulness_score(answer, context)
    return {
        "chunk_relevancy": relevancy,
        "faithfulness":    faithful,
    }
