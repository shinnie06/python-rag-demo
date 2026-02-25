"""
All prompt templates for the MCU RAG application.
Centralised here for easy tuning and experimentation.
"""
from __future__ import annotations


# ── Shared system persona ─────────────────────────────────────────────────────

MCU_SYSTEM = (
    "You are MarvelMind, an expert AI assistant specialised in the Marvel Cinematic Universe (MCU). "
    "You answer questions based ONLY on the provided context documents. "
    "Always cite your sources using [Source: <title/filename>] inline. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be concise, accurate, and engaging."
)


# ── Query rewriting ───────────────────────────────────────────────────────────

QUERY_REWRITE_TEMPLATE = """\
You are a query optimisation assistant for a Marvel MCU knowledge base.
Rewrite the following user query to improve document retrieval accuracy.
Make the rewritten query more specific, include relevant MCU terminology, and remove ambiguity.
Output ONLY the rewritten query — no explanation, no quotes.

Original query: {query}
Rewritten query:"""


# ── QA with citations ─────────────────────────────────────────────────────────

QA_TEMPLATE = """\
Answer the following question about the Marvel Cinematic Universe using ONLY the context below.
Cite each piece of information with its source in brackets, e.g. [Source: Iron Man Wikipedia].
If the context does not contain the answer, say: "I don't have enough information in my knowledge base to answer this."

Context:
{context}

Question: {question}

Answer:"""


# ── Character profile builder ─────────────────────────────────────────────────

CHARACTER_PROFILE_TEMPLATE = """\
Using ONLY the context documents below, create a structured profile for the Marvel character "{character}".

Include these sections (skip any not covered by the context):
- **Real Name & Alias**
- **Powers & Abilities**
- **First Appearance** (film or comic)
- **Key Films / Appearances**
- **Key Relationships** (allies, enemies)
- **Notable Storylines or Moments**
- **Actor** (if mentioned)

Cite each fact with [Source: <title>].

Context:
{context}

Profile:"""


# ── Timeline narrator ─────────────────────────────────────────────────────────

TIMELINE_TEMPLATE = """\
Using ONLY the context documents below, answer this MCU timeline question clearly and chronologically.
Present events in order where possible. Cite sources with [Source: <title>].

Context:
{context}

Timeline question: {question}

Answer:"""


# ── Quiz question generator ───────────────────────────────────────────────────

QUIZ_GEN_TEMPLATE = """\
You are a Marvel MCU trivia quiz master. Using ONLY the context documents below, generate {num_questions} multiple-choice trivia questions.

Rules:
- Each question must be directly answerable from the context
- Provide 4 options (A, B, C, D) — only one is correct
- Include a brief explanation of the correct answer citing the source
- Vary difficulty: mix easy, medium, and hard questions
- Return ONLY valid JSON in this exact format (no markdown fences):

[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "... [Source: ...]"
  }}
]

Context:
{context}

Topic hint: {topic}

JSON quiz:"""


# ── No-RAG baseline (for RAG vs No-RAG comparison) ────────────────────────────

NO_RAG_TEMPLATE = """\
Answer the following question about the Marvel Cinematic Universe using your general knowledge only.
Be honest if you are uncertain.

Question: {question}

Answer:"""


# ── Faithfulness judge ────────────────────────────────────────────────────────

FAITHFULNESS_JUDGE_TEMPLATE = """\
You are an objective evaluator. Rate how faithfully the following answer is grounded in the provided context.

Scoring guide:
5 = Every claim in the answer is directly supported by the context
4 = Most claims are supported; minor extrapolations
3 = About half the claims are supported
2 = Few claims are supported; significant hallucination
1 = The answer contradicts or ignores the context

Context:
{context}

Answer to evaluate:
{answer}

Output ONLY a single integer between 1 and 5. No explanation.
Score:"""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string with source labels.
    chunks: list of dicts with 'text' and 'metadata' keys.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = (
            meta.get("title")
            or meta.get("filename")
            or meta.get("character")
            or meta.get("source", f"Document {i}")
        )
        parts.append(f"[Source: {source}]\n{chunk['text'].strip()}")
    return "\n\n---\n\n".join(parts)
