"""
MCU Fandom Wiki scraper via the MediaWiki REST API.
No JavaScript rendering needed — the API returns plain text directly.

Source: https://marvelcinematicuniverse.fandom.com
API:    https://marvelcinematicuniverse.fandom.com/api.php
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

from llama_index.core import Document

from mcu_rag.config import FANDOM_ARTICLES, FANDOM_DIR

FANDOM_API = "https://marvelcinematicuniverse.fandom.com/api.php"
FANDOM_BASE = "https://marvelcinematicuniverse.fandom.com/wiki/"


def _slug(title: str) -> str:
    """Convert article title to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:80]


def _clean_wikitext(raw: str) -> str:
    """
    Strip common wikitext markup and emit structured Markdown.
    Headings are preserved as ## / ### / #### so MarkdownNodeParser
    can split at section boundaries during ingestion.
    """
    # Remove templates {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", " ", raw)
    # Remove file/image links [[File:...]]
    text = re.sub(r"\[\[File:[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Image:[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Convert [[link|display]] → display; [[link]] → link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove external links [http... text] → text
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove bold/italic markup
    text = re.sub(r"'''?", "", text)
    # Convert wikitext headings → Markdown headings (deepest first to avoid partial matches)
    text = re.sub(r"====([^=\n]+)====", lambda m: f"\n\n#### {m.group(1).strip()}\n", text)
    text = re.sub(r"===([^=\n]+)===",   lambda m: f"\n\n### {m.group(1).strip()}\n", text)
    text = re.sub(r"==([^=\n]+)==",     lambda m: f"\n\n## {m.group(1).strip()}\n", text)
    # Remove horizontal rules
    text = re.sub(r"^----+$", "", text, flags=re.MULTILINE)
    # Remove reference tags
    text = re.sub(r"<ref[^/]*/?>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", " ", text)
    # Clean up excess whitespace
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_article(title: str, session: requests.Session) -> str | None:
    """
    Fetch a single Fandom wiki article via the MediaWiki API.
    Returns the cleaned article text, or None if the page doesn't exist.
    """
    params = {
        "action":    "query",
        "titles":    title,
        "redirects": "1",        # follow redirects (e.g. "Iron Man" → "Tony Stark/Iron Man")
        "prop":      "revisions",
        "rvprop":    "content",
        "rvslots":   "main",
        "format":    "json",
    }
    try:
        resp = session.get(FANDOM_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  FETCH ERROR ({title}): {exc}")
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None  # page not found
        try:
            raw = page["revisions"][0]["slots"]["main"]["*"]
            return _clean_wikitext(raw)
        except (KeyError, IndexError):
            return None
    return None


def scrape_fandom_articles(
    titles: list[str] | None = None,
    sleep_sec: float = 0.8,
) -> list[Document]:
    """
    Scrape MCU Fandom wiki articles.
    Caches each article as a .txt file in FANDOM_DIR to avoid re-fetching.
    Returns LlamaIndex Documents.
    """
    FANDOM_DIR.mkdir(parents=True, exist_ok=True)
    targets = titles or FANDOM_ARTICLES

    session = requests.Session()
    session.headers.update({
        "User-Agent": "MCU-RAG-Demo/1.0 (educational project; contact: student@example.com)"
    })

    documents: list[Document] = []
    fetched = 0
    cached  = 0

    for title in targets:
        cache_path = FANDOM_DIR / f"{_slug(title)}.txt"

        # Use cache if available
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            cached += 1
            article_status = "cached"
        else:
            text = _fetch_article(title, session)
            if not text or len(text) < 50:
                print(f"  SKIP (empty/not found): {title}")
                continue
            cache_path.write_text(text, encoding="utf-8")
            fetched += 1
            article_status = "fetched"
            time.sleep(sleep_sec)   # polite rate limiting

        url = FANDOM_BASE + title.replace(" ", "_")
        # Prepend article title as h1 so MarkdownNodeParser has full heading context
        md_text = f"# {title}\n\n{text}"
        doc = Document(
            text=md_text,
            metadata={
                "source":  "fandom_wiki",
                "title":   title,
                "url":     url,
                "type":    "wiki_article",
            },
        )
        documents.append(doc)
        print(f"  Fandom [{article_status}]: {title} ({len(text):,} chars)")

    print(f"  Fandom total: {len(documents)} articles ({cached} cached, {fetched} newly fetched)")
    return documents


def load_fandom_documents() -> list[Document]:
    """Load all configured MCU Fandom wiki articles."""
    print(f"  Fetching {len(FANDOM_ARTICLES)} Fandom wiki articles...")
    return scrape_fandom_articles()
