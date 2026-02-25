"""
BM25 keyword retriever.
Builds/loads an index over all ingested chunks for exact keyword matching.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from mcu_rag.config import BM25_INDEX_PATH, BM25_TOP_K


def _tokenise(text: str) -> list[str]:
    """
    Whitespace + punctuation tokeniser, lowercased.
    Preserves hyphenated compound names (e.g. 'Spider-Man' → 'spider-man')
    and acronyms by keeping hyphens and dots within tokens.
    """
    text = text.lower()
    # Normalise acronyms: S.H.I.E.L.D. → shield
    text = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", text)
    # Match tokens including internal hyphens (spider-man, ant-man, etc.)
    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text)
    return tokens


class BM25Retriever:
    """
    Wrapper around rank-bm25's BM25Okapi.
    Stores the raw chunk texts and metadata alongside the index for retrieval.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list[dict] = []  # list of {text, metadata}

    def build(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts (must have 'text' key).
        chunks: [{"text": "...", "metadata": {...}}, ...]
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as e:
            raise ImportError("Install 'rank-bm25': uv add rank-bm25") from e

        self._chunks = chunks
        tokenised = [_tokenise(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        print(f"  BM25 index built over {len(chunks)} chunks.")

    def save(self, path: Path | None = None) -> None:
        """Serialise the index to disk."""
        target = path or BM25_INDEX_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
        print(f"  BM25 index saved to {target}")

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved index from disk."""
        target = path or BM25_INDEX_PATH
        if not target.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {target}. "
                "Run: uv run python scripts/ingest.py"
            )
        with open(target, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        """
        Search for top_k chunks matching the query.
        Returns list of dicts: {text, metadata, score, rank}.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built/loaded yet.")

        tokens = _tokenise(query)
        raw_scores = self._bm25.get_scores(tokens)

        # Get top_k indices by score
        top_indices = sorted(
            range(len(raw_scores)),
            key=lambda i: raw_scores[i],
            reverse=True,
        )[:top_k]

        results = []
        max_score = raw_scores[top_indices[0]] if top_indices else 1.0
        for rank, idx in enumerate(top_indices):
            score = raw_scores[idx]
            if score <= 0:
                break
            results.append({
                "text":     self._chunks[idx]["text"],
                "metadata": self._chunks[idx].get("metadata", {}),
                "score":    round(float(score) / max(max_score, 1e-9), 4),
                "raw_score": float(score),
                "rank":     rank + 1,
            })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None


# Module-level singleton — loaded lazily
_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """Return the module-level BM25Retriever, loading from disk on first call."""
    global _retriever
    if _retriever is None or not _retriever.is_loaded:
        _retriever = BM25Retriever()
        _retriever.load()
    return _retriever
