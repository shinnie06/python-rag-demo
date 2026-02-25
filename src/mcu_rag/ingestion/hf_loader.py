"""
Load and normalise HuggingFace MCU datasets into LlamaIndex Documents.
Supports: Manvith/Marvel_dataset, ismaildlml/Jarvis-MCU-Dialogues, rohitsaxena/MovieSum
"""
from __future__ import annotations

from typing import Any

from llama_index.core import Document

from mcu_rag.config import HF_DATASETS, MCU_FILM_TITLES, RAW_DIR


def _safe_str(val: Any) -> str:
    """Convert any value to a clean string, handling NaN/None."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "") else s


# ── Per-dataset row converters ────────────────────────────────────────────────

def _row_to_text_marvel(row: dict) -> tuple[str, dict] | None:
    """Manvith/Marvel_dataset — character info with debut/actor data."""
    parts = []
    name = _safe_str(row.get("name") or row.get("Name") or row.get("character_name") or "")

    for k, v in row.items():
        text_v = _safe_str(v)
        if text_v:
            label = k.replace("_", " ").title()
            parts.append(f"{label}: {text_v}")

    text = f"MCU Character — {name}\n" + "\n".join(parts) if name else "\n".join(parts)
    return text, {
        "source":    "huggingface_marvel",
        "character": name,
        "type":      "character",
    }


def _row_to_text_dialogues(row: dict) -> tuple[str, dict] | None:
    """ismaildlml/Jarvis-MCU-Dialogues — Tony Stark / Jarvis dialogue pairs."""
    speaker  = _safe_str(row.get("speaker") or row.get("Speaker") or "")
    dialogue = _safe_str(row.get("dialogue") or row.get("text") or row.get("line") or "")
    film     = _safe_str(row.get("movie") or row.get("film") or row.get("source") or "MCU")

    if not dialogue:
        # Fallback: dump all non-empty fields
        parts = [f"{k}: {_safe_str(v)}" for k, v in row.items() if _safe_str(v)]
        text = "\n".join(parts)
    else:
        text = f"[{film}] {speaker}: {dialogue}" if speaker else f"[{film}] {dialogue}"

    return text, {
        "source":    "huggingface_dialogues",
        "character": speaker,
        "film":      film,
        "type":      "dialogue",
    }


def _row_to_text_moviesum(row: dict) -> tuple[str, dict] | None:
    """
    rohitsaxena/MovieSum — film screenplay + Wikipedia summary pairs.
    Only MCU films are kept (filtered by MCU_FILM_TITLES).
    Returns None for non-MCU films.
    """
    title = _safe_str(row.get("title") or row.get("movie_title") or row.get("name") or "")

    # Filter: only keep MCU films
    if title.lower() not in MCU_FILM_TITLES:
        return None

    summary  = _safe_str(row.get("summary") or row.get("wikipedia_summary") or row.get("plot") or "")
    synopsis = _safe_str(row.get("synopsis") or row.get("screenplay") or "")

    parts = [f"Film: {title}"]
    if summary:
        parts.append(f"Summary: {summary}")
    if synopsis:
        parts.append(f"Synopsis: {synopsis[:2000]}")   # cap screenplay length

    text = "\n\n".join(parts)
    return text, {
        "source": "huggingface_moviesum",
        "film":   title,
        "type":   "film_summary",
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_HANDLERS = {
    "marvel":    _row_to_text_marvel,
    "dialogues": _row_to_text_dialogues,
    "moviesum":  _row_to_text_moviesum,
}


def load_hf_documents() -> list[Document]:
    """
    Download all configured HuggingFace datasets and return LlamaIndex Documents.
    Each row becomes one Document with source metadata.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError("Install 'datasets': uv add datasets") from e

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    for ds_config in HF_DATASETS:
        ds_name  = ds_config["name"]
        split    = ds_config.get("split", "train")
        handler  = ds_config.get("handler", "marvel")
        convert  = _HANDLERS.get(handler, _row_to_text_marvel)

        print(f"  Loading {ds_name} [{split}] (handler={handler})...")

        try:
            dataset = load_dataset(ds_name, split=split, trust_remote_code=True)
        except Exception as exc:
            print(f"  WARNING: Could not load {ds_name}: {exc}")
            continue

        rows = dataset.to_pandas().to_dict(orient="records")
        loaded = 0
        skipped = 0

        for i, row in enumerate(rows):
            result = convert(row)
            if result is None:           # filtered out (e.g. non-MCU MovieSum row)
                skipped += 1
                continue

            text, meta = result
            if len(text.strip()) < 20:   # skip near-empty rows
                skipped += 1
                continue

            meta["row_index"] = i
            meta["dataset"]   = ds_name
            documents.append(Document(text=text, metadata=meta))
            loaded += 1

        print(f"  -> {loaded} documents loaded, {skipped} skipped from {ds_name}")

    print(f"  Total HuggingFace documents: {len(documents)}")
    return documents
