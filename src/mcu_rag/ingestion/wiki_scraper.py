"""
Scrape Wikipedia MCU articles and convert them to LlamaIndex Documents.
Caches scraped text to data/wikipedia/ to avoid re-fetching.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from llama_index.core import Document

from mcu_rag.config import WIKI_CHARACTERS, WIKI_DIR, WIKI_FILMS, WIKI_PAGES


def _clean_text(text: str) -> str:
    """Remove excessive whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _page_to_markdown(page) -> str:
    """
    Convert a wikipediaapi page to structured Markdown.
    Uses page.sections to emit ## / ### headings so MarkdownNodeParser
    can split at section boundaries during ingestion.
    """
    parts: list[str] = [f"# {page.title}"]

    # Lead / intro section (text before the first heading)
    summary = _clean_text(getattr(page, "summary", "") or "")
    if summary:
        parts.append(summary)

    def _add_sections(sections: list, level: int = 2) -> None:
        for section in sections:
            body = _clean_text(section.text)
            if not body and not section.sections:
                continue  # skip empty / stub sections
            parts.append(f"{'#' * level} {section.title}")
            if body:
                parts.append(body)
            _add_sections(section.sections, min(level + 1, 6))

    _add_sections(page.sections)
    return "\n\n".join(parts)


def _slug(title: str) -> str:
    """Convert article title to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:80]


def scrape_wiki_articles(
    titles: list[str],
    language: str = "en",
    sleep_sec: float = 0.5,
) -> list[Document]:
    """
    Fetch Wikipedia articles for each title.
    Returns LlamaIndex Documents with metadata.
    Cached: if the .txt file already exists in WIKI_DIR it is reused.
    """
    try:
        import wikipediaapi  # type: ignore
    except ImportError as e:
        raise ImportError("Install 'Wikipedia-API': uv add Wikipedia-API") from e

    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    wiki = wikipediaapi.Wikipedia(
        language=language,
        user_agent="MCU-RAG-Demo/1.0 (educational project)",
    )

    documents: list[Document] = []

    for title in titles:
        cache_path = WIKI_DIR / f"{_slug(title)}.txt"

        # --- Use cache if available ---
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        else:
            page = wiki.page(title)
            if not page.exists():
                print(f"  SKIP (not found): {title}")
                continue

            text = _page_to_markdown(page)
            url = page.fullurl

            if len(text) < 100:
                print(f"  SKIP (too short): {title}")
                continue

            cache_path.write_text(text, encoding="utf-8")
            time.sleep(sleep_sec)  # polite rate limiting

        doc = Document(
            text=text,
            metadata={
                "source": "wikipedia",
                "title": title,
                "url": url,
                "type": "article",
            },
        )
        documents.append(doc)
        print(f"  Wikipedia: {title} ({len(text):,} chars)")

    return documents


def load_wiki_documents() -> list[Document]:
    """Load all configured Wikipedia articles (characters + films + lore pages)."""
    all_titles = WIKI_CHARACTERS + WIKI_FILMS + WIKI_PAGES
    print(f"  Fetching {len(all_titles)} Wikipedia articles...")
    docs = scrape_wiki_articles(all_titles)
    print(f"  Total Wikipedia documents: {len(docs)}")
    return docs
