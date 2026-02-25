"""
Load user-uploaded PDF files into LlamaIndex Documents.
Uses Docling when available (superior table/layout extraction);
falls back to LlamaIndex SimpleDirectoryReader (pypdf) when Docling is not installed.

Install Docling:  uv add docling
Without Docling:  works out of the box with existing dependencies.
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core import Document

from mcu_rag.config import UPLOADS_DIR


def _has_docling() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def _load_with_docling(pdf_files: list[Path]) -> list[Document]:
    """Parse PDFs using Docling — preserves tables, headings, and multi-column layouts."""
    from docling.document_converter import DocumentConverter  # type: ignore

    converter = DocumentConverter()
    documents = []

    for pdf_path in pdf_files:
        try:
            result = converter.convert(str(pdf_path))
            # Export to Markdown — best format for RAG chunking
            text = result.document.export_to_markdown()
            if not text.strip():
                continue
            documents.append(Document(
                text=text,
                metadata={
                    "source":   "pdf_upload",
                    "filename": pdf_path.name,
                    "type":     "uploaded_pdf",
                    "parser":   "docling",
                },
            ))
        except Exception as exc:
            print(f"  WARNING: Docling failed on {pdf_path.name}: {exc}. Trying fallback...")
            docs = _load_single_fallback(pdf_path)
            documents.extend(docs)

    return documents


def _load_with_fallback(pdf_files: list[Path]) -> list[Document]:
    """Parse PDFs using LlamaIndex SimpleDirectoryReader (pypdf backend)."""
    from llama_index.core import SimpleDirectoryReader  # type: ignore

    documents = []
    for pdf_path in pdf_files:
        docs = _load_single_fallback(pdf_path)
        documents.extend(docs)
    return documents


def _load_single_fallback(pdf_path: Path) -> list[Document]:
    """Load one PDF via SimpleDirectoryReader and enrich metadata."""
    from llama_index.core import SimpleDirectoryReader  # type: ignore

    try:
        reader = SimpleDirectoryReader(input_files=[str(pdf_path)])
        raw_docs = reader.load_data()
    except Exception as exc:
        print(f"  WARNING: Could not parse {pdf_path.name}: {exc}")
        return []

    for doc in raw_docs:
        doc.metadata.update({
            "source":   "pdf_upload",
            "filename": pdf_path.name,
            "type":     "uploaded_pdf",
            "parser":   "pypdf",
        })
    return raw_docs


# ── Public API ────────────────────────────────────────────────────────────────

def load_pdf_documents(directory: Path | None = None) -> list[Document]:
    """
    Load all PDFs from the given directory (defaults to data/uploads/).
    Uses Docling if installed, otherwise falls back to pypdf.
    Returns an empty list if the directory has no PDFs.
    """
    target = directory or UPLOADS_DIR
    target.mkdir(parents=True, exist_ok=True)

    pdf_files = list(target.glob("*.pdf"))
    if not pdf_files:
        print("  No PDF files found in uploads/ — skipping.")
        return []

    use_docling = _has_docling()
    parser_name = "Docling" if use_docling else "pypdf (install docling for better quality)"
    print(f"  Loading {len(pdf_files)} PDF(s) from {target} using {parser_name}...")

    documents = (
        _load_with_docling(pdf_files)
        if use_docling
        else _load_with_fallback(pdf_files)
    )

    print(f"  Total PDF documents: {len(documents)}")
    return documents


def ingest_single_pdf(pdf_path: Path) -> list[Document]:
    """
    Load a single PDF. Used for live ingestion from the Streamlit UI.
    Uses Docling if installed, otherwise falls back to pypdf.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    use_docling = _has_docling()
    if use_docling:
        docs = _load_with_docling([pdf_path])
    else:
        docs = _load_single_fallback(pdf_path)

    if not docs:
        raise RuntimeError(f"No content extracted from {pdf_path.name}")
    return docs
