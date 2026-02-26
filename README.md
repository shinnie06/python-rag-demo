# MarvelMind — Advanced RAG Deep Dive

> **An AI course mini project** that showcases every layer of a production-grade Retrieval-Augmented Generation (RAG) system. Built to teach, built to demo. Uses the Marvel Cinematic Universe as the knowledge domain — rich, fun, and with a built-in proof of RAG's value.

Fully **local** — no API keys, no cloud, no cost after setup.

---

## Table of Contents

1. [What Problem Does RAG Solve?](#1-what-problem-does-rag-solve)
2. [Why the MCU Is the Perfect Demo Domain](#2-why-the-mcu-is-the-perfect-demo-domain)
3. [Architecture at a Glance](#3-architecture-at-a-glance)
4. [The Knowledge Base — Building the Foundation](#4-the-knowledge-base--building-the-foundation)
5. [Step 1: Chunking — Splitting Documents Intelligently](#5-step-1-chunking--splitting-documents-intelligently)
6. [Step 2: Embedding — Turning Text into Numbers](#6-step-2-embedding--turning-text-into-numbers)
7. [Step 3: Indexing — Two Indices, Two Strengths](#7-step-3-indexing--two-indices-two-strengths)
8. [Step 4: Query Rewriting + Multi-Query Expansion](#8-step-4-query-rewriting--multi-query-expansion)
9. [Step 5: Hybrid Retrieval — BM25 + Vector Search](#9-step-5-hybrid-retrieval--bm25--vector-search)
10. [Step 6: Reciprocal Rank Fusion — Merging Two Result Lists](#10-step-6-reciprocal-rank-fusion--merging-two-result-lists)
11. [Step 7: CrossEncoder Reranking — The Quality Gate](#11-step-7-crossencoder-reranking--the-quality-gate)
12. [Step 8: Prompt Assembly — Grounding the LLM](#12-step-8-prompt-assembly--grounding-the-llm)
13. [Step 9: LLM Generation — Streaming the Answer](#13-step-9-llm-generation--streaming-the-answer)
14. [Step 10: Evaluation — Measuring Quality](#14-step-10-evaluation--measuring-quality)
15. [The App Pages](#15-the-app-pages)
16. [Project Structure](#16-project-structure)
17. [Setup & Running](#17-setup--running)
18. [Data Sources](#18-data-sources)
19. [Configuration Reference](#19-configuration-reference)
20. [Demo Script for Assessors](#20-demo-script-for-assessors)

---

## 1. What Problem Does RAG Solve?

### The Core Problem: LLMs Have a Knowledge Cutoff

Large language models are trained on text data collected up to a certain date. After that date, the model has no knowledge of new events, new releases, new information. This is called the **knowledge cutoff**.

When you ask a question about something beyond the cutoff, the model has two bad options:

- **Say "I don't know"** — unhelpful
- **Hallucinate** — invent a plausible-sounding but wrong answer

Neither is acceptable in production.

### The Naive Solution: Just Fine-tune the Model

You could retrain or fine-tune the model with new data. But this is:
- Extremely expensive (millions of dollars for a large model)
- Slow (weeks or months)
- Brittle (every update requires retraining)
- Risky (fine-tuning can cause the model to "forget" other knowledge)

### The RAG Solution: Give the Model the Information at Query Time

**Retrieval-Augmented Generation (RAG)** takes a completely different approach:

1. Build a searchable knowledge base from your documents
2. When a user asks a question, search the knowledge base for relevant passages
3. Include those passages in the prompt to the LLM
4. Ask the LLM to answer *using the provided context*, not its training data

The LLM no longer needs to "know" the answer — it just needs to read the passages you give it and synthesize a response. This is called **grounding**: the answer is grounded in retrieved evidence.

```
Without RAG:
  User: "Who plays Mr. Fantastic in the new MCU film?"
  LLM: "I don't have information about this." (or worse: hallucinates)

With RAG:
  User: "Who plays Mr. Fantastic in the new MCU film?"
  Retrieved: "Pedro Pascal was cast as Reed Richards / Mr. Fantastic in
              The Fantastic Four: First Steps, released July 2025."
  LLM: "Pedro Pascal plays Mr. Fantastic (Reed Richards) in The Fantastic
        Four: First Steps (July 2025)."
```

### Why RAG Beats Fine-Tuning for Most Problems

| | Fine-tuning | RAG |
|--|-------------|-----|
| Update cost | Retrain entire model | Update knowledge base |
| Update speed | Weeks | Minutes |
| Verifiable sources | No | Yes (citations) |
| Hallucination risk | Still present | Significantly reduced |
| Works with private data | Risky (bakes data in) | Yes (data never leaves DB) |
| Cost | $$$$ | $ |

RAG is the dominant approach for enterprise applications precisely because it is cheap, updatable, and verifiable.

---

## 2. Why the MCU Is the Perfect Demo Domain

### The Knowledge Cutoff Is Visible

`llama3.1:8b` — the local LLM powering this app — was trained on data through **December 2023**.

MCU films and shows released after that date are completely unknown to the model:

| Release | Date | LLM Knows? |
|---------|------|------------|
| *Deadpool & Wolverine* | Aug 2024 | No |
| *Agatha All Along* | Sep 2024 | No |
| *Captain America: Brave New World* | Feb 2025 | No |
| *Daredevil: Born Again* | Mar 2025 | No |
| *Thunderbolts\** | May 2025 | No |
| *The Fantastic Four: First Steps* | Jul 2025 | No |
| *Avengers: Doomsday* | 2026 | No |

This creates a **natural controlled experiment**:

- Ask about Tony Stark → both RAG and No-RAG know the answer (LLM was trained on this)
- Ask about Pedro Pascal as Mr. Fantastic → RAG answers correctly; No-RAG hallucinates or fails
- Ask about the plot of Thunderbolts* → RAG retrieves the Fandom wiki article; No-RAG invents something plausible but wrong

The MCU domain is also **assessor-friendly**: people who evaluate the project know the domain, can instantly spot hallucinations, and find the subject matter engaging.

### Built-In Ground Truth

Post-cutoff MCU content has objective correct answers. "Who directed Thunderbolts*?" has one correct answer. This makes evaluation trivially easy — if the model gets it right, RAG worked; if it hallucinates, it failed.

---

## 3. Architecture at a Glance

The full pipeline from user question to grounded answer:

```
╔══════════════════════════════════════════════════════════════════════╗
║                     OFFLINE (run once)                               ║
║                                                                      ║
║  Wikipedia ──┐                                                       ║
║  Fandom Wiki ─┤ Scrape → Markdown → MarkdownNodeParser → Chunks     ║
║  HuggingFace ─┤ Load  → Normalise → SentenceSplitter  → Chunks     ║
║  User PDFs  ──┘ Read  → Extract  → SentenceSplitter  → Chunks     ║
║                              │                                       ║
║                    ┌─────────┴─────────┐                            ║
║                    ▼                   ▼                             ║
║             Embed with            Tokenise                           ║
║          nomic-embed-text         (BM25)                             ║
║                    │                   │                             ║
║                    ▼                   ▼                             ║
║             ChromaDB             bm25_index.pkl                     ║
║          (vector store)          (keyword index)                     ╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                      ONLINE (per query)                              ║
║                                                                      ║
║  User Query                                                          ║
║      │                                                               ║
║      ▼                                                               ║
║  [1] Query Rewriting ──── LLM rewrites query; expands into 3        ║
║      │                    variants searched independently            ║
║      ▼                                                               ║
║  [2] BM25 Search ──────── Exact keyword matching (top 10)           ║
║  [3] Vector Search ─────── Semantic similarity search (top 10)      ║
║      │  (parallel)                                                   ║
║      ▼                                                               ║
║  [4] RRF Fusion ────────── Merge both result lists into one         ║
║      │                    ranked list (deduplication included)       ║
║      ▼                                                               ║
║  [5] CrossEncoder ─────── Neural model re-scores every              ║
║      │                    (query, chunk) pair → top 5               ║
║      ▼                                                               ║
║  [6] Prompt Assembly ───── System prompt + retrieved chunks         ║
║      │                    + source citations assembled               ║
║      ▼                                                               ║
║  [7] LLM Generation ────── llama3.1:8b generates grounded           ║
║      │                    streaming answer                           ║
║      ▼                                                               ║
║  [8] Evaluation ─────────── Chunk relevancy + Faithfulness          ║
║                             scores computed and displayed            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 4. The Knowledge Base — Building the Foundation

The knowledge base is the foundation of the entire system. Everything depends on having high-quality, well-structured documents to retrieve from.

### What Goes In

Four complementary sources are combined:

#### Wikipedia (95 articles)
Scraped via the `wikipedia-api` library. Covers MCU characters, films, and lore. Critically, the Wikipedia scraper converts articles into **structured Markdown** — preserving section headings as `## Section`, `### Subsection` — so the chunking strategy can split at semantic boundaries.

```python
# wiki_scraper.py: _page_to_markdown()
# Traverses page.sections recursively to emit structured headings
def _page_to_markdown(page) -> str:
    parts = [f"# {page.title}"]
    # Lead / intro
    parts.append(_clean_text(page.summary))
    # Recursive section traversal
    def _add_sections(sections, level=2):
        for section in sections:
            parts.append(f"{'#' * level} {section.title}")
            parts.append(_clean_text(section.text))
            _add_sections(section.sections, min(level + 1, 6))
    _add_sections(page.sections)
    return "\n\n".join(parts)
```

#### MCU Fandom Wiki (53 articles)
Scraped via the **MediaWiki REST API** — no JavaScript rendering required, no scraping brittle HTML. Returns raw wikitext which is then cleaned and converted to Markdown:

```python
# fandom_scraper.py: _clean_wikitext()
# Converts wikitext headings to Markdown headings (deepest-first to avoid partial matches)
text = re.sub(r"====([^=\n]+)====", lambda m: f"\n\n#### {m.group(1).strip()}\n", text)
text = re.sub(r"===([^=\n]+)===",   lambda m: f"\n\n### {m.group(1).strip()}\n", text)
text = re.sub(r"==([^=\n]+)==",     lambda m: f"\n\n## {m.group(1).strip()}\n", text)
```

Why Fandom over just Wikipedia? Fandom wiki covers **in-universe** detail that Wikipedia often lacks — exact dialogue, plot minutiae, character relationships, post-credits scenes. For questions like "What exactly happened in the end credits of Captain America: Brave New World?", Fandom is far more detailed.

#### HuggingFace Datasets (2 datasets)

| Dataset | Content | Why included |
|---------|---------|--------------|
| `Manvith/Marvel_dataset` | Character stats, actor names, debut appearances | Structured factual lookups |
| `ismaildlml/Jarvis-MCU-Dialogues` | FRIDAY/JARVIS in-universe AI dialogue | Conversational MCU voice |

> `rohitsaxena/MovieSum` was removed — the dataset returned 0 MCU documents after filtering (title column format mismatch, all 1,800 rows skipped). Film plot summaries are fully covered by Wikipedia and Fandom film articles.

Each HuggingFace dataset row is normalised into a LlamaIndex `Document` with metadata: `{source, character, film, phase, year}`.

#### User PDF Uploads
Any PDF uploaded at runtime via the Home page sidebar is ingested live — chunked, embedded, added to ChromaDB and BM25. No restart required. This demonstrates that RAG systems are **mutable, live knowledge bases**, not static indexes.

### Metadata: The Hidden Power

Every chunk is stored with metadata tags:

```python
{
    "source":    "fandom_wiki",      # where it came from
    "title":     "Avengers: Endgame",# article/document title
    "url":       "https://...",      # source URL for citation
    "type":      "wiki_article",     # article type
}
```

Metadata enables **filtered retrieval** — e.g., "only retrieve chunks from Phase 5 films" or "only retrieve from Wikipedia". This is how the QA Chat sidebar filters work.

---

## 5. Step 1: Chunking — Splitting Documents Intelligently

### Why Chunking Matters

Language models have a **context window** — a maximum number of tokens they can process at once. `llama3.1:8b` has an 8K token context window. A Wikipedia article about Tony Stark can be 50,000+ tokens. We cannot embed or retrieve the entire article as one unit.

Chunking splits documents into smaller passages (chunks) that:
1. Fit within the embedding model's context window
2. Are semantically coherent (contain one complete idea)
3. Are small enough to be precisely relevant to a specific query

### The Wrong Way to Chunk

The naive approach is to split every N characters:

```
"Iron Man is a superhero appearing in American... [512 chars]
...ess suit. In 2008, the film Iron Man [512 chars]
...k Stark. He founded Stark Industries [512 chars]"
```

Problem: chunk boundaries are arbitrary. A chunk might start mid-sentence, split a key fact across two chunks, or mix content from two unrelated sections.

### The Right Way: Source-Aware Chunking

This project uses **two different chunking strategies** depending on the source:

#### MarkdownNodeParser for Wikipedia and Fandom Wiki

```
# Tony Stark

Tony Stark is a fictional superhero...

## Early Life
Howard and Maria Stark's son, Anthony Edward Stark was born...

## Iron Man Armor
The Iron Man armor is a powered exoskeleton that provides...

## Relationships
### Pepper Potts
Tony and Pepper's relationship began as employer/employee...
```

`MarkdownNodeParser` splits at heading boundaries:
- Chunk 1: Everything under `# Tony Stark` (the intro)
- Chunk 2: Everything under `## Early Life`
- Chunk 3: Everything under `## Iron Man Armor`
- Chunk 4: Everything under `## Relationships / ### Pepper Potts`

Each chunk stays **semantically coherent** — it's a complete section about one topic.

#### SentenceSplitter for HuggingFace Datasets and PDFs

Flat, unstructured text has no headings to split on. `SentenceSplitter` uses sentence boundaries:
- Never cuts in the middle of a sentence
- `chunk_size=512` tokens, `chunk_overlap=64` tokens
- The overlap ensures context isn't lost at chunk boundaries

#### Handling Oversized Sections

Some Wikipedia sections (e.g., Tony Stark's "Biography" section) are enormous — 5000+ tokens. A MarkdownNodeParser chunk that big would waste embedding context and retrieve too much irrelevant text.

The solution: any Markdown node exceeding `CHUNK_SIZE × 4` characters is re-wrapped as a Document and sub-split by `SentenceSplitter`:

```python
# vector_store.py: _chunk_documents()
_max_chars = CHUNK_SIZE * 4  # ~2048 chars ≈ 512 tokens

for node in raw_nodes:
    if len(node.get_content()) > _max_chars:
        # Re-wrap as Document → sub-split at sentence boundaries
        oversized_docs.append(Document(text=node.get_content(), metadata=node.metadata))
    else:
        all_nodes.append(node)

if oversized_docs:
    all_nodes.extend(sentence_splitter.get_nodes_from_documents(oversized_docs))
```

### Why This Dual Strategy Is Better

| Strategy | Best for | Why |
|----------|---------|-----|
| MarkdownNodeParser | Structured articles (Wikipedia, Fandom) | Preserves section context; "Iron Man Powers" stays together |
| SentenceSplitter | Flat text (datasets, PDFs, dialogues) | No headings to split on; sentence boundaries are the next best |

**The result:** when you ask "What are Tony Stark's Iron Man armors?", the retrieved chunk is the entire "Iron Man Armor" section — not a 512-token window that happens to start in the middle of a different topic.

---

## 6. Step 2: Embedding — Turning Text into Numbers

### What Is an Embedding?

An **embedding** is a dense vector of floating-point numbers that represents the *meaning* of a piece of text. Two texts that mean similar things will have vectors that point in similar directions in high-dimensional space.

```
"Tony Stark built the Iron Man suit"  → [0.23, -0.41, 0.87, 0.12, ...]  (768 dims)
"The Iron Man armor was created by Tony"  → [0.21, -0.39, 0.85, 0.14, ...]  (similar!)
"Thanos collected the Infinity Stones"  → [-0.15, 0.72, -0.33, 0.91, ...]  (very different)
```

The closer two vectors are (measured by cosine similarity or L2 distance), the more semantically similar the texts.

### Why Use Local Embeddings?

This project uses `nomic-embed-text` via Ollama — a local embedding model running entirely on your machine.

Advantages:
- **No API cost** — embed unlimited documents for free
- **Privacy** — your documents never leave your machine
- **8K token context** — can embed long sections without truncation
- **Fast** — ~80 embeddings/second on Apple Silicon

### How Embeddings Enable Search

When a user submits a query, the system:
1. Embeds the query into a vector using the same model
2. Searches ChromaDB for stored chunk vectors that are "close" to the query vector
3. Returns the closest chunks — the ones most semantically similar to the query

This enables **semantic search**: queries and documents don't need to share exact words. "Who created Iron Man's suit?" will still retrieve chunks about "Tony Stark built the armor" even though "created" and "built" are different words — their meanings are similar in embedding space.

### Similarity Score

ChromaDB returns L2 (Euclidean) distance between query and chunk vectors. We convert this to a similarity score:

```python
# vector_store.py
score = 1.0 / (1.0 + dist)   # ∈ (0, 1], higher = more similar
```

Using `1/(1+d)` rather than `1 - d/2` ensures the score is always positive and never clips valid results, regardless of how large the L2 distance is.

---

## 7. Step 3: Indexing — Two Indices, Two Strengths

The project builds two separate indices over the same corpus of chunks. They have complementary strengths.

### Index 1: ChromaDB Vector Store

**What it is:** A persistent database of chunk embeddings. Built and maintained by ChromaDB, an open-source vector database.

**What it's good at:** Semantic / conceptual similarity. Finds documents that *mean* the same thing as the query, even if they use different words.

**Weakness:** Struggles with exact term matching. A query for "S.H.I.E.L.D." might not retrieve chunks that mention the organisation if the embedding model generalises too heavily.

**Built during ingestion:** `build_vector_index()` takes all chunk nodes, embeds them via `nomic-embed-text`, and stores them in ChromaDB at `data/chroma_db/`.

### Index 2: BM25 Keyword Index

**What it is:** A serialised `rank-bm25` index. BM25 (Best Match 25) is a classic information retrieval algorithm — the same family of algorithms that powers Elasticsearch.

**What it's good at:** Exact term matching. Finds documents that *contain* the same words as the query. Perfect for named entities: "Thanos", "Mjolnir", "Wakanda", specific dates, specific film titles.

**Weakness:** Blind to meaning. A query for "Tony Stark's weapon" will miss chunks about "Iron Man's repulsor" even though they refer to the same thing, because the words don't match.

**Tokenisation matters:** The tokeniser preserves hyphenated compound names:

```python
# bm25_retriever.py: _tokenise()
# Preserves: "Spider-Man" → "spider-man" (one token, not two)
# Normalises: "S.H.I.E.L.D." → "shield" (acronym normalisation)
tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text.lower())
```

Without this, "Spider-Man" would become `["spider", "man"]` — losing the compound entity. With it, `"spider-man"` is one token that matches the same token in chunks.

**Built during ingestion:** After ChromaDB is populated, the ingest script fetches all chunks back from ChromaDB, tokenises them, and builds a BM25Okapi index. It serialises the index to `data/bm25_index.pkl` so it doesn't need to be rebuilt on every startup.

---

## 8. Step 4: Query Rewriting + Multi-Query Expansion

### The Problem

Users write conversational queries, not search queries:

```
User types:       "Who's that guy with the hammer again?"
Good for search:  "Thor Odinson hammer Mjolnir MCU character"
```

Vector and BM25 search both work better with explicit, noun-heavy queries. A second, subtler problem: a single rewrite that goes wrong (e.g. hallucinating a wrong year or dropping a subtitle) poisons the entire retrieval pipeline — there is no fallback.

### Step A: Hallucination-Resistant Primary Rewrite

The pipeline rewrites the query using strict rules that prevent the LLM from inventing facts:

```
STRICT RULES — violating these ruins retrieval:
1. PRESERVE all proper nouns EXACTLY: movie titles (including subtitles),
   character names, actor names. Do NOT add years/dates not stated by the user.
2. DO NOT invent facts not in the original query.
3. You may expand vague pronouns or abbreviations using MCU terminology.
4. Add retrieval-useful terms only when clearly implied.
5. Output ONE line — the rewritten query only.
```

Few-shot examples steer the model toward the correct behaviour:

| Original | Rewritten |
|---|---|
| "Who's the spider guy and what can he do?" | "Who is Spider-Man Peter Parker MCU powers abilities" |
| "What did RDJ do in Endgame?" | "What did Robert Downey Jr Tony Stark do in Avengers: Endgame?" |
| "Post credit scene in 'The Fantastic Four: First Steps'?" | "post-credits scene The Fantastic Four: First Steps MCU Phase 6 implications" |

**Safety checks:**
- Length bounds: rewrite must be 5–500 characters, else fall back to original
- **Quoted-phrase guard**: if the user quoted a title (e.g. `'The Fantastic Four: First Steps'`) and key words from that phrase do not appear in the rewrite, the pipeline falls back to the original query — preventing dropped subtitles or paraphrased entity names
- Temperature raised from 0.0 → 0.3 to reduce over-committed wrong paraphrases

### Step B: Multi-Query Expansion

A single rewrite, even a good one, can still miss relevant chunks. The pipeline generates **2 additional query variants** that retrieve from different angles:

```
Variant 1 (keyword-dense):   strips conversational words, maximises BM25 recall
Variant 2 (semantic paraphrase): different phrasing of same intent, better for vector search
```

All 3 queries — primary rewrite + 2 variants — run the **full hybrid pipeline independently** (BM25 → Vector → RRF → CrossEncoder). The resulting chunk pools are merged and deduplicated before final CrossEncoder reranking. This means the system can recover from a single bad rewrite via the other variants.

```
Original query
    │
    ▼
Primary rewrite ──── hybrid_retrieve() ─── chunks A
    │
    ├── Variant 1 (keyword-dense) ──────── hybrid_retrieve() ─── chunks B
    │
    └── Variant 2 (semantic paraphrase) ── hybrid_retrieve() ─── chunks C
                                                                      │
                                                         merge + dedup + re-sort
                                                                      │
                                                            CrossEncoder final top-K
```

### Why Not Use the Rewritten Query for the Answer?

The rewritten queries are only used for **retrieval** — finding relevant chunks. The original user query is used for **generation** — the LLM still sees what the user actually asked, ensuring the final answer addresses the question as phrased.

---

## 9. Step 5: Hybrid Retrieval — BM25 + Vector Search

### Running Both Retrievers in Parallel

The rewritten query is sent to both indices simultaneously:

```
Rewritten query: "Thor Mjolnir hammer Asgard MCU"
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      BM25 Search               Vector Search
    (keyword match)           (semantic similarity)
    top 10 results              top 10 results
```

**BM25 results** (from `bm25_retriever.py`): Chunks scored by BM25Okapi formula. Chunks containing "Thor", "Mjolnir", "hammer", or "Asgard" rank highly. A chunk specifically mentioning "Mjolnir the enchanted hammer" scores very high.

**Vector results** (from `vector_store.py`): Chunks with embeddings closest to the query embedding. Chunks about "Thor's power", "Asgardian weapons", "worthy heroes" score well — even if they don't mention "Mjolnir" explicitly.

### Why Both Are Necessary

Consider a query: *"What did Tony say before the snap?"*

- **BM25 alone**: Looks for chunks with words "Tony", "snap". Gets chunks about Thanos's snap, but might miss dialogue chunks that don't say "snap" explicitly.
- **Vector alone**: Understands "snap" semantically relates to Infinity War's climax. Gets semantically related chunks, but might miss the exact dialogue chunk.
- **Both**: BM25 finds exact-match chunks; Vector finds semantic matches; together they cover more ground.

Real-world evidence from IR research shows hybrid consistently outperforms either method alone, typically by 10-20% on recall metrics.

---

## 10. Step 6: Reciprocal Rank Fusion — Merging Two Result Lists

### The Problem

After BM25 and Vector search, we have two lists of 10 results each — but they have incompatible scores:

- BM25 scores: raw term frequency scores (e.g., 0.0 to 12.4)
- Vector scores: similarity scores (e.g., 0.42 to 0.91)

We can't simply average these — they're on completely different scales. We need a way to merge two ranked lists into one that respects both orderings.

### Reciprocal Rank Fusion (RRF)

RRF solves this elegantly. Instead of using raw scores, it uses **rank position**:

```
RRF score for chunk C = Σ  1 / (k + rank_in_list_i)
                       i=1
```

where `k = 60` (a constant that dampens the influence of very high-ranked items) and `rank` starts at 1.

**Example:**

| Chunk | BM25 Rank | Vector Rank | RRF Score |
|-------|-----------|-------------|-----------|
| Chunk A | 1 | 3 | 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = **0.0323** |
| Chunk B | 2 | — | 1/(60+2) = **0.0161** |
| Chunk C | — | 1 | 1/(60+1) = **0.0164** |
| Chunk D | 5 | 2 | 1/(60+5) + 1/(60+2) = 0.0154 + 0.0161 = **0.0315** |

**Chunks that appear in both lists get a significant boost** — they are deduped and their scores are summed. This is intentional: if both keyword search AND semantic search agree a chunk is relevant, it almost certainly is.

### Why k=60?

The constant `k=60` was established in the original RRF paper (Cormack et al., 2009) through empirical testing across many IR benchmarks. It controls how much the algorithm penalises lower-ranked items. A smaller k gives more weight to top-ranked items; a larger k flattens the distribution. `k=60` has proven robust across many retrieval scenarios.

### The Result

After RRF, we have one merged, deduplicated list of candidates ranked by their combined evidence from both retrieval methods. Typically 10-18 unique candidates (since some chunks appear in both lists and get merged).

---

## 11. Step 7: CrossEncoder Reranking — The Quality Gate

### The Problem with Bi-Encoder Search

Both BM25 and vector search are what information retrieval researchers call **bi-encoder** approaches: the query and document are encoded independently and then compared.

```
Query embedding:    [0.23, -0.41, 0.87, ...]
Document embedding: [0.21, -0.39, 0.85, ...]
Score: cosine_similarity(query_emb, doc_emb) = 0.97
```

This is fast — you pre-compute all document embeddings and only embed the query at search time. But it's imprecise: the query and document never "see" each other. The model can't reason about how well the *specific words* of the document address the *specific question* being asked.

### CrossEncoder: Query and Document Together

A **CrossEncoder** reads the query AND the document together in a single forward pass:

```
Input:  [CLS] Who built the Iron Man suit? [SEP] Tony Stark, a genius billionaire
        inventor, designed and built the Iron Man armor in a cave... [SEP]

Output: Score 9.2  (high — directly answers the question)
```

vs.

```
Input:  [CLS] Who built the Iron Man suit? [SEP] The Avengers assembled in
        New York to fight the Chitauri invasion led by Loki... [SEP]

Output: Score -3.1  (low — tangentially related to MCU but doesn't answer)
```

The CrossEncoder can see exactly how the question relates to the passage. It can detect when a chunk is about the right topic but doesn't actually answer the specific question.

### The Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

This model was fine-tuned on the **MS MARCO** passage ranking dataset — millions of real search queries and human-judged relevant passages. It's small (22M parameters), fast (runs on CPU), and downloads automatically on first use.

### The Trade-off

CrossEncoder is too slow to run over the entire knowledge base — it can't pre-compute scores because it needs the query. But it's very accurate when run over a small candidate set.

**The two-stage approach (common in production RAG):**

```
Stage 1: Fast bi-encoder search (BM25 + vector) → top 20 candidates
Stage 2: Slow CrossEncoder → re-score and rerank → top 5 final results

Speed: O(1) for Stage 1 (pre-computed embeddings), O(n) for Stage 2 (but n=20, not 7000)
Quality: Much better than Stage 1 alone
```

### CrossEncoder Score Interpretation

The raw CrossEncoder scores have no fixed range — they can be negative or large positive numbers. What matters is their relative ordering: higher score = more relevant.

In the RAG Insights bar chart:
- **Blue bars** = positive scores (the chunk directly addresses the query)
- **Red bars** = negative scores (poor match, shouldn't be in the final results)
- **Position in list = final rank** (after CrossEncoder reordering)

The before/after comparison in RAG Insights shows how often CrossEncoder changes the order from RRF — typically 2-4 chunks move significantly. This demonstrates that RRF alone is not sufficient.

---

## 12. Step 8: Prompt Assembly — Grounding the LLM

### From Chunks to Context

The top 5 chunks returned by CrossEncoder are assembled into a context string:

```python
# prompts.py: format_context()
def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("title") or meta.get("source", "Knowledge Base")
        parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
```

Each chunk is labelled with its source so the LLM can cite it in the answer.

### The Prompt Template

```
You are an expert on the Marvel Cinematic Universe. Answer questions
using ONLY the information provided in the context below. If the context
doesn't contain enough information to answer fully, say so — do not
invent details.

For each factual claim, cite the source in brackets, e.g. [Source 1].

CONTEXT:
[Source 1: The Fantastic Four: First Steps]
Pedro Pascal was cast as Reed Richards / Mr. Fantastic in The Fantastic
Four: First Steps. The film is set in a retro-futuristic 1960s universe...

[Source 2: Reed Richards (Wikipedia)]
Reed Richards, also known as Mr. Fantastic, is a superhero in Marvel Comics...

---

QUESTION: Who plays Mr. Fantastic in the new MCU film?
```

### Why the System Prompt Matters

The instruction "using ONLY the information provided in the context" is critical. Without it, the LLM will blend retrieved context with its training data — sometimes correctly, but sometimes hallucinating details that aren't in the context.

The instruction forces the LLM into "reading comprehension mode" rather than "knowledge recall mode". It must answer from the provided text, not from memory.

---

## 13. Step 9: LLM Generation — Streaming the Answer

### The Local LLM: llama3.1:8b via Ollama

[Ollama](https://ollama.com) runs open-source LLMs locally with a simple API. `llama3.1:8b` is Meta's LLaMA 3.1 model at 8 billion parameters — powerful enough for nuanced MCU questions, small enough to run on a MacBook Pro with ~8GB VRAM/memory.

### Streaming vs Blocking

Users experience streaming output — tokens appear one by one as they're generated, like watching someone type. This creates a much better UX than waiting 10-30 seconds for the full response.

```python
# llm.py: stream_generate() — yields tokens one at a time
stream = client.chat(model=model, messages=messages, stream=True, ...)
for chunk in stream:
    token = chunk.get("message", {}).get("content", "")
    if token:
        yield token
```

Streamlit's `st.write_stream()` consumes this generator and renders tokens live.

### Error Handling

If Ollama is not running, the LLM wrapper raises a `RuntimeError` with a clear message:
```
"Ollama streaming failed — is Ollama running? (model: llama3.1:8b)"
```

This surfaces cleanly in the Streamlit UI as an error message rather than a mysterious crash.

---

## 14. Step 10: Evaluation — Measuring Quality

After generation, the system automatically evaluates the response quality using two metrics.

### Metric 1: Chunk Relevancy

**Question answered:** "Are the retrieved chunks actually about what the user asked?"

**Method:** Cosine similarity between the query embedding and each retrieved chunk embedding, averaged across all chunks.

```python
# scorer.py: chunk_relevancy_score()
query_emb = embed_model.get_text_embedding(query)
chunk_embs = [embed_model.get_text_embedding(c["text"]) for c in chunks]
similarities = [_cosine_similarity(query_emb, emb) for emb in chunk_embs]
return float(np.mean(similarities))
```

**Interpretation:**
- 0.8+ = Excellent. Retrieved chunks are highly relevant.
- 0.6-0.8 = Good. Most chunks are on-topic.
- Below 0.5 = Concern. Retrieval may have failed; answer could be unreliable.

**Limitation:** A chunk can be semantically similar to the query but still not answer it. High relevancy doesn't guarantee a correct answer — it means retrieval found topically related content.

### Metric 2: Faithfulness (LLM-as-Judge)

**Question answered:** "Is the generated answer grounded in the retrieved context, or is the LLM hallucinating?"

**Method:** The LLM judges its own answer:

```
System: You are an evaluation judge. Score how well the ANSWER is supported
        by the CONTEXT on a scale of 1-5.

        5 = Every claim in the answer is directly supported by the context
        3 = Most claims are supported; minor additions from general knowledge
        1 = The answer contradicts or ignores the provided context

CONTEXT: {retrieved chunks}
ANSWER: {generated answer}

Output only a single integer from 1 to 5.
```

The raw score (1-5) is normalised to (0-1) for display.

**Why LLM-as-Judge?**
- No human annotation required
- Fast (one extra LLM call)
- Surprisingly reliable for detecting hallucinations
- The same LLM generated the answer AND evaluates it — this can be biased, but it's a practical local alternative to expensive external evaluators like RAGAS (which require OpenAI API)

**Interpretation:**
- 0.8+ = High faithfulness. Answer sticks closely to retrieved context.
- 0.5-0.8 = Moderate. Some claims may go beyond the provided context.
- Below 0.5 = Low faithfulness. The LLM may be hallucinating. Don't trust the answer.

---

## 15. The App Pages

### 🏠 Home
The entry point shows **system status** (Ollama running? Embeddings available? ChromaDB has chunks?), a **visual pipeline diagram** of all 7 steps, and a **PDF upload panel** for live knowledge base enrichment.

### 💬 QA Chat
Free-form MCU question answering with:
- **Streaming token-by-token output** — answers appear live
- **Sidebar filters** — narrow retrieval to a specific source (`wikipedia`, `fandom_wiki`, `huggingface_marvel`, `pdf_upload`) or MCU Phase (1–6)
- **RAG Trace** — expandable panel showing the rewritten query + all retrieved chunks with their scores
- **Citations** — source list under every answer
- **Evaluation scores** — Chunk Relevancy + Faithfulness displayed after each response

### 🧠 Quiz Mode
Enter any topic → the pipeline retrieves relevant chunks → the LLM generates multiple-choice trivia questions *from those specific chunks*. Every question is grounded in retrieved evidence, not invented from training data. After submission, the chunks that inspired the questions are revealed.

### 🦸 Character Dive
Select any MCU character → one click builds a comprehensive, source-cited profile using the character RAG prompt template. If no chunks are found for the character, a warning is shown so you know the answer is ungrounded.

### 📅 Timeline Explorer
Ask temporal MCU questions: *"What happened between Infinity War and Endgame?"*, *"Summarise Phase 5"*, *"What led to the formation of the Avengers?"*. A Phase Reference table (Phases 1–6) is shown for context. The timeline prompt template instructs the LLM to organise its answer chronologically.

### 🔬 RAG Insights (the centrepiece)

The educational deep-dive page. Run any query and see the entire pipeline live:

| Step | What you see |
|------|-------------|
| **Chunking Explainer** | Side-by-side: SentenceSplitter output vs MarkdownNodeParser output for the same article |
| **Step 1: Query Rewriting** | Original → primary rewrite → 2 multi-query variants; prompts shown; all 3 queries hit the vector DB independently |
| **Step 2: Retrieval Comparison** | BM25 results / Vector results / Hybrid side by side; 🔗 marks chunks that appeared in both lists |
| **Step 3: CrossEncoder Reranking** | Bar chart of all merged candidates scored by CrossEncoder; Before vs After position changes with 🔼🔽 arrows |
| **Step 4: Final Answer** | Generated answer with sources |
| **Step 5: Evaluation** | Chunk Relevancy + Faithfulness gauges with explanations |
| **RAG vs No-RAG** | Toggle: same question answered with and without retrieved context — the most powerful demo for post-cutoff queries |

**The "Why do some results look irrelevant?" explainer** in Step 2 is particularly educational: it explains BM25 false positives (keyword match without semantic match) and why Vector search can retrieve topically adjacent but not directly relevant chunks.

---

## 16. Project Structure

```
mcu-rag/
├── pyproject.toml                   # uv dependencies
├── .python-version                  # Python 3.13
├── README.md                        # This document
├── Home.py                          # Streamlit entry point
│
├── pages/
│   ├── 1_💬_QA_Chat.py             # Streaming Q&A + citations + RAG trace
│   ├── 2_🧠_Quiz_Mode.py           # LLM-generated MCU trivia
│   ├── 3_🦸_Character_Dive.py      # Per-character RAG profile
│   ├── 4_📅_Timeline_Explorer.py   # Temporal MCU queries
│   └── 5_🔬_RAG_Insights.py        # Educational pipeline trace viewer
│
├── src/
│   └── mcu_rag/
│       ├── config.py                # ← Single source of truth for all constants
│       ├── ingestion/
│       │   ├── hf_loader.py         # HuggingFace → Documents
│       │   ├── wiki_scraper.py      # Wikipedia → Markdown → Documents
│       │   ├── fandom_scraper.py    # Fandom Wiki → Markdown → Documents
│       │   └── pdf_loader.py        # PDF → Documents
│       ├── retrieval/
│       │   ├── vector_store.py      # ChromaDB: build / load / query / add
│       │   ├── bm25_retriever.py    # BM25: build / save / load / search
│       │   └── hybrid.py            # BM25 + Vector → RRF → CrossEncoder
│       ├── generation/
│       │   ├── llm.py               # Ollama streaming wrapper
│       │   ├── prompts.py           # All prompt templates
│       │   └── pipeline.py          # Orchestrator: rewrite → retrieve → generate → eval
│       └── evaluation/
│           └── scorer.py            # Chunk relevancy + faithfulness scoring
│
├── scripts/
│   └── ingest.py                    # CLI: build ChromaDB + BM25 from all sources
│
└── data/
    ├── wikipedia/                   # Cached Wikipedia .txt (Markdown format)
    ├── fandom/                      # Cached Fandom wiki .txt (Markdown format)
    ├── uploads/                     # User-uploaded PDFs
    └── chroma_db/                   # Persisted ChromaDB vector store
```

### Key Design Principles

**`config.py` as single source of truth:** All model names, paths, and hyperparameters live in one file. Change `LLM_MODEL = "qwen2.5:7b"` and the entire app switches models without touching any other file.

**Lazy-loading singletons:** The CrossEncoder, BM25 index, and RAG pipeline are all loaded once and cached for the lifetime of the Streamlit session. The first query has some startup latency; subsequent queries are fast.

**Graceful degradation:** If BM25 fails, the pipeline falls back to vector-only. If CrossEncoder fails, it falls back to RRF-ordered results. If Ollama is down, a clear error message surfaces. The app never crashes silently.

---

## 17. Setup & Running

### Prerequisites

1. **[Ollama](https://ollama.com)** — local LLM server
2. **[uv](https://docs.astral.sh/uv/)** — Python package manager

```bash
# Install Ollama (macOS)
brew install ollama

# Pull required models
ollama pull llama3.1:8b        # LLM for generation + evaluation (~4.7GB)
ollama pull nomic-embed-text   # Embedding model (~274MB)

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
uv sync
```

### Build the Knowledge Base

```bash
# Full ingestion — recommended on first run (~10-20 min depending on network)
uv run python scripts/ingest.py

# Flags:
uv run python scripts/ingest.py --skip-wiki      # skip Wikipedia scraping
uv run python scripts/ingest.py --skip-fandom    # skip Fandom wiki scraping
uv run python scripts/ingest.py --skip-hf        # skip HuggingFace datasets
uv run python scripts/ingest.py --reset          # wipe and rebuild from scratch
```

Expected result: **~7,200+ chunks** indexed in ChromaDB, BM25 index saved.

### Run the App

```bash
uv run streamlit run Home.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## 18. Data Sources

| Source | Articles | Content | Chunking |
|--------|---------|---------|---------|
| Wikipedia | ~103 | Characters, films, Phase 4–6 Disney+ shows, MCU lore | MarkdownNodeParser |
| MCU Fandom Wiki | ~90 | In-universe detail, plot summaries, events, objects | MarkdownNodeParser |
| `Manvith/Marvel_dataset` | 39 docs | Character stats, actors, debut appearances | SentenceSplitter |
| `ismaildlml/Jarvis-MCU-Dialogues` | 601 docs | In-universe JARVIS/FRIDAY dialogue | SentenceSplitter |
| User PDFs | Variable | Any MCU content you upload | SentenceSplitter |

> `rohitsaxena/MovieSum` removed — returned 0 MCU docs (title format mismatch). Film coverage is fully provided by Wikipedia and Fandom.

**Post-cutoff content in the knowledge base** (not in LLM's training data — ideal for RAG demo):

- **Phase 4 Disney+ shows:** WandaVision, The Falcon and the Winter Soldier, Loki, Hawkeye, Ms. Marvel, Moon Knight, She-Hulk: Attorney at Law
- **Phase 5 shows:** Secret Invasion, Echo, Agatha All Along, Daredevil: Born Again
- **Phase 5 films:** Deadpool & Wolverine, Captain America: Brave New World, Thunderbolts\*
- **Phase 6:** The Fantastic Four: First Steps, Avengers: Doomsday
- **Characters:** Cassandra Nova, Valentina Allegra de Fontaine, Bob Reynolds (Sentry), Matthew Murdock, Thaddeus Ross (Red Hulk), Reed Richards / Mr. Fantastic (Pedro Pascal), Sue Storm, Johnny Storm, Ben Grimm, Victor von Doom

---

## 19. Configuration Reference

All constants in [`src/mcu_rag/config.py`](src/mcu_rag/config.py):

```python
# Models
LLM_MODEL           = "llama3.1:8b"
EMBED_MODEL         = "nomic-embed-text"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE    = 512    # tokens per chunk for SentenceSplitter
CHUNK_OVERLAP = 64     # token overlap between consecutive chunks (preserves context at boundaries)

# Retrieval
BM25_TOP_K    = 10     # candidates from BM25 keyword search
VECTOR_TOP_K  = 10     # candidates from vector semantic search
RERANK_TOP_K  = 5      # final chunks passed to the LLM after CrossEncoder
```

**Tuning tips:**
- Increase `RERANK_TOP_K` to 8-10 for complex questions that need more context
- Decrease `CHUNK_SIZE` to 256 for more precise retrieval (more chunks, each more specific)
- Swap `LLM_MODEL` to `qwen2.5:7b` for a different generation quality/speed trade-off

---

## 20. Demo Script for Assessors

### The One Query to Show First

Open RAG Insights and run: **"Who plays Mr. Fantastic in The Fantastic Four: First Steps?"**

Toggle **RAG vs No-RAG**:
- Without RAG: The LLM says it doesn't know (or hallucinates a wrong actor)
- With RAG: "Pedro Pascal plays Reed Richards / Mr. Fantastic in The Fantastic Four: First Steps (July 2025)." — with citation

This single demo proves the entire value proposition of RAG in 30 seconds.

### Recommended Demo Flow (10 minutes)

1. **Home** (1 min) — show Ollama running locally, 7,200+ chunks, no API keys
2. **RAG Insights** (4 min) — run the Mr. Fantastic query
   - Show RAG vs No-RAG toggle
   - Walk through Step 2 (BM25 vs Vector vs Hybrid comparison)
   - Walk through Step 3 (CrossEncoder before/after reranking)
3. **QA Chat** (2 min) — ask "What is the plot of Thunderbolts*?" — expand RAG trace
4. **Quiz Mode** (1 min) — generate 5 questions on "Agatha All Along"
5. **PDF Upload** (2 min) — upload any MCU PDF, ask a question about it

### Other Strong Demo Queries

| Query | Why it's powerful |
|-------|-----------------|
| "Who is Cassandra Nova?" | Main villain of Deadpool & Wolverine (Aug 2024) — LLM blind spot |
| "What happened in Agatha All Along?" | Sep 2024 Disney+ show — pure RAG territory |
| "What is the Red Hulk?" | Phase 5 character reveal — perfect hallucination vs RAG comparison |
| "Tell me about the Snap" | Phase 3 — LLM knows this; shows high faithfulness score |
| "What led to the Battle of New York?" | Tests multi-chunk retrieval across Wikipedia + Fandom sources |

### Key Talking Points

- **"Why not just ChatGPT?"** — This runs 100% locally. Your data never leaves your machine. Works offline after setup. No monthly API bill.
- **"Why hybrid search?"** — Show the BM25/Vector comparison in RAG Insights. Different queries favour different methods. Hybrid wins consistently.
- **"Why CrossEncoder?"** — Show the before/after reranking. A chunk about "Iron Man" that mentions Thor tangentially might rank #1 after BM25+Vector but drop to #4 after CrossEncoder correctly identifies it doesn't answer the specific question.
- **"Why MCU?"** — The knowledge cutoff creates an objective, observable proof of RAG's value. Old MCU content = LLM knows it. New MCU content = LLM is blind. RAG bridges the gap.

---

## Dependencies

```toml
streamlit>=1.40          # UI framework
chromadb>=0.5            # Vector store (persistent, local)
llama-index-core>=0.11   # RAG framework; MarkdownNodeParser, SentenceSplitter, Document
llama-index-vector-stores-chroma   # ChromaDB ↔ LlamaIndex integration
llama-index-embeddings-ollama      # nomic-embed-text via Ollama
llama-index-llms-ollama            # llama3.1:8b via Ollama
llama-index-readers-file           # PDF loading
rank-bm25>=0.2.2         # BM25Okapi keyword search index
sentence-transformers>=3.0 # CrossEncoder ms-marco-MiniLM-L-6-v2
datasets>=2.20           # HuggingFace dataset loading
wikipedia-api>=0.6       # Wikipedia article scraping
requests>=2.32           # Fandom wiki MediaWiki API requests
tqdm>=4.66               # Ingestion progress bars
numpy>=1.26              # Cosine similarity computation
pandas>=2.2              # Dataset manipulation
```

---

*MarvelMind — Built for an AI course mini project showcase.*
*Powered by Ollama + ChromaDB + LlamaIndex + Streamlit.*
*All Marvel content referenced for educational purposes.*
