"""
One-time data ingestion script.
Run once (or after --reset) to build ChromaDB + BM25 from all data sources.

Usage:
    python scripts/ingest.py                   # full ingestion
    python scripts/ingest.py --skip-hf         # skip HuggingFace datasets
    python scripts/ingest.py --skip-wiki       # skip Wikipedia scraping
    python scripts/ingest.py --skip-fandom     # skip Fandom wiki scraping
    python scripts/ingest.py --reset           # wipe and rebuild from scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # type: ignore

from mcu_rag.config import BM25_INDEX_PATH, CHROMA_COLLECTION, CHROMA_DIR
from mcu_rag.ingestion.fandom_scraper import load_fandom_documents
from mcu_rag.ingestion.hf_loader import load_hf_documents
from mcu_rag.ingestion.pdf_loader import load_pdf_documents
from mcu_rag.ingestion.wiki_scraper import load_wiki_documents
from mcu_rag.retrieval.bm25_retriever import BM25Retriever
from mcu_rag.retrieval.vector_store import build_vector_index, get_chroma_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCU RAG ingestion pipeline")
    parser.add_argument("--skip-hf",     action="store_true", help="Skip HuggingFace datasets")
    parser.add_argument("--skip-wiki",   action="store_true", help="Skip Wikipedia scraping")
    parser.add_argument("--skip-fandom", action="store_true", help="Skip Fandom wiki scraping")
    parser.add_argument("--reset",       action="store_true", help="Delete existing index and rebuild")
    return parser.parse_args()


def reset_index() -> None:
    """Delete the existing ChromaDB collection and BM25 index."""
    print("  Resetting existing indices...")
    try:
        client = get_chroma_client()
        client.delete_collection(CHROMA_COLLECTION)
        print("  ChromaDB collection deleted.")
    except Exception:
        pass

    if BM25_INDEX_PATH.exists():
        BM25_INDEX_PATH.unlink()
        print("  BM25 index deleted.")


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  MarvelMind — MCU RAG Ingestion Pipeline")
    print("=" * 60)
    print("\nData sources:")
    print(f"  HuggingFace datasets : {'SKIP' if args.skip_hf else 'ON'}")
    print(f"  Wikipedia scraping   : {'SKIP' if args.skip_wiki else 'ON'}")
    print(f"  Fandom wiki scraping : {'SKIP' if args.skip_fandom else 'ON'}")
    print(f"  PDF uploads          : ON (data/uploads/)")

    if args.reset:
        reset_index()

    # ── 1. Load all documents ──────────────────────────────────────────────
    all_documents = []

    if not args.skip_hf:
        print("\n[1/5] Loading HuggingFace datasets...")
        hf_docs = load_hf_documents()
        all_documents.extend(hf_docs)
    else:
        print("\n[1/5] Skipping HuggingFace datasets (--skip-hf)")

    if not args.skip_wiki:
        print("\n[2/5] Scraping Wikipedia articles...")
        wiki_docs = load_wiki_documents()
        all_documents.extend(wiki_docs)
    else:
        print("\n[2/5] Skipping Wikipedia scraping (--skip-wiki)")

    if not args.skip_fandom:
        print("\n[3/5] Scraping MCU Fandom wiki articles...")
        fandom_docs = load_fandom_documents()
        all_documents.extend(fandom_docs)
    else:
        print("\n[3/5] Skipping Fandom wiki scraping (--skip-fandom)")

    print("\n[4/5] Loading uploaded PDFs...")
    pdf_docs = load_pdf_documents()
    all_documents.extend(pdf_docs)

    if not all_documents:
        print("\nERROR: No documents loaded. Check your data sources.")
        sys.exit(1)

    print(f"\n  Total documents to index: {len(all_documents)}")

    # ── 2. Build vector index (ChromaDB) ──────────────────────────────────
    print("\n[5/5] Building ChromaDB vector index (this may take several minutes)...")
    build_vector_index(all_documents)

    # ── 3. Build BM25 index from the same chunks ──────────────────────────
    print("\n  Building BM25 index from ChromaDB chunks...")
    client = get_chroma_client()
    collection = client.get_collection(CHROMA_COLLECTION)
    total_chunks = collection.count()

    print(f"  Fetching {total_chunks} chunks for BM25...")

    batch_size = 500
    all_chunk_dicts = []
    offset = 0
    with tqdm(total=total_chunks, desc="  Fetching chunks") as pbar:
        while offset < total_chunks:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            batch_docs  = result.get("documents", [])
            batch_metas = result.get("metadatas", [])

            for text, meta in zip(batch_docs, batch_metas):
                if text:
                    all_chunk_dicts.append({"text": text, "metadata": meta or {}})

            fetched = len(batch_docs)
            offset += fetched
            pbar.update(fetched)
            if fetched < batch_size:
                break

    bm25 = BM25Retriever()
    bm25.build(all_chunk_dicts)
    bm25.save()

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Ingestion complete!")
    print(f"  Documents loaded : {len(all_documents)}")
    print(f"  Vector chunks    : {total_chunks}")
    print(f"  BM25 chunks      : {len(all_chunk_dicts)}")
    print(f"  ChromaDB path    : {CHROMA_DIR}")
    print(f"  BM25 index path  : {BM25_INDEX_PATH}")
    print("=" * 60)
    print("\n  Run the app with:")
    print("  streamlit run Home.py\n")


if __name__ == "__main__":
    main()
