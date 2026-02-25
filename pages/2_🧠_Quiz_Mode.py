"""
Quiz Mode — LLM generates MCU trivia from the knowledge base, user is scored.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcu_rag.generation.pipeline import get_pipeline

st.set_page_config(page_title="Quiz Mode — MarvelMind", page_icon="🧠", layout="wide")

st.title("🧠 MCU Trivia Quiz")
st.caption(
    "Questions are generated in real-time from the knowledge base — "
    "every question is grounded in actual retrieved documents."
)

# ── Session state init ────────────────────────────────────────────────────────
if "quiz_questions"  not in st.session_state: st.session_state.quiz_questions  = []
if "quiz_answers"    not in st.session_state: st.session_state.quiz_answers    = {}  # q_idx -> chosen option
if "quiz_revealed"   not in st.session_state: st.session_state.quiz_revealed   = {}  # q_idx -> bool
if "quiz_score"      not in st.session_state: st.session_state.quiz_score      = 0
if "quiz_submitted"  not in st.session_state: st.session_state.quiz_submitted  = False
if "quiz_topic"      not in st.session_state: st.session_state.quiz_topic      = ""
if "quiz_chunks"     not in st.session_state: st.session_state.quiz_chunks     = []

# ── Sidebar: quiz setup ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quiz Settings")

    topic = st.text_input(
        "Topic / character / film",
        placeholder="e.g. Thanos, Avengers: Endgame, Phase 3",
        value="",
    )
    num_q = st.slider("Number of questions", min_value=3, max_value=10, value=5)

    generate_btn = st.button("Generate Quiz", type="primary", use_container_width=True)

    if st.session_state.quiz_score > 0 or st.session_state.quiz_submitted:
        st.divider()
        total = len(st.session_state.quiz_questions)
        score = st.session_state.quiz_score
        pct   = int(score / max(total, 1) * 100)
        st.markdown(f"**Score: {score}/{total} ({pct}%)**")
        if pct >= 80:
            st.success("Excellent! You're an MCU expert!")
        elif pct >= 60:
            st.info("Good effort! Keep watching those films.")
        else:
            st.warning("Keep studying — Thanos would be disappointed.")

    if st.button("Reset Quiz", use_container_width=True):
        for key in ["quiz_questions", "quiz_answers", "quiz_revealed", "quiz_score", "quiz_submitted", "quiz_topic", "quiz_chunks"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Generate quiz ─────────────────────────────────────────────────────────────
if generate_btn:
    effective_topic = topic.strip() or "Marvel Cinematic Universe general knowledge"

    with st.spinner("Generating quiz questions from knowledge base..."):
        pipeline = get_pipeline()
        result = pipeline.run(
            query=effective_topic,
            mode="quiz",
            topic=effective_topic,
            num_questions=num_q,
        )
        raw_json = result["answer"]
        chunks   = result["chunks"]

    # Parse JSON
    questions = []
    try:
        # Strip markdown fences if present
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            raw_json = "\n".join(raw_json.split("\n")[1:])
        if raw_json.endswith("```"):
            raw_json = "\n".join(raw_json.split("\n")[:-1])
        questions = json.loads(raw_json.strip())
    except json.JSONDecodeError:
        st.error("Could not parse quiz JSON. Try a different topic or regenerate.")
        st.code(raw_json, language="json")

    if questions:
        st.session_state.quiz_questions = questions
        st.session_state.quiz_answers   = {}
        st.session_state.quiz_revealed  = {}
        st.session_state.quiz_score     = 0
        st.session_state.quiz_submitted = False
        st.session_state.quiz_topic     = effective_topic
        st.session_state.quiz_chunks    = chunks
        st.rerun()

# ── Render quiz ────────────────────────────────────────────────────────────────
questions = st.session_state.quiz_questions

if not questions:
    st.info("Use the sidebar to generate a quiz on any MCU topic.")
    st.markdown("**Example topics:**")
    examples_cols = st.columns(3)
    example_topics = [
        "Iron Man", "Thanos and the Infinity Gauntlet", "Avengers: Endgame",
        "Spider-Man", "Doctor Strange and the Multiverse", "Black Panther and Wakanda",
    ]
    for i, t in enumerate(example_topics):
        col = examples_cols[i % 3]
        if col.button(t, key=f"qt_{t}"):
            st.session_state.quiz_topic = t
            st.rerun()
    st.stop()

st.markdown(f"### Topic: *{st.session_state.quiz_topic}*")
st.markdown(f"**{len(questions)} questions generated from your knowledge base**")
st.divider()

# Render each question
for i, q in enumerate(questions):
    q_text   = q.get("question", f"Question {i+1}")
    options  = q.get("options", {})
    correct  = q.get("correct", "A")
    explain  = q.get("explanation", "")

    st.markdown(f"**Q{i+1}.** {q_text}")

    chosen = st.radio(
        f"q{i}",
        options=list(options.keys()),
        format_func=lambda k, opts=options: f"{k}) {opts[k]}",
        key=f"radio_{i}",
        label_visibility="collapsed",
        disabled=st.session_state.quiz_submitted,
    )
    st.session_state.quiz_answers[i] = chosen

    # After submission: show correct/wrong + explanation
    if st.session_state.quiz_submitted:
        if chosen == correct:
            st.success(f"Correct! ({correct})")
        else:
            st.error(f"Wrong. Correct answer: {correct}) {options.get(correct, '')}")

        if explain:
            with st.expander("Explanation"):
                st.markdown(explain)


    st.divider()

# ── Submit button ─────────────────────────────────────────────────────────────
if not st.session_state.quiz_submitted:
    all_answered = len(st.session_state.quiz_answers) == len(questions)
    if st.button("Submit Quiz", type="primary", disabled=not all_answered):
        score = sum(
            1 for i, q in enumerate(questions)
            if st.session_state.quiz_answers.get(i) == q.get("correct")
        )
        st.session_state.quiz_score     = score
        st.session_state.quiz_submitted = True
        st.rerun()

    if not all_answered:
        remaining = len(questions) - len(st.session_state.quiz_answers)
        st.caption(f"Answer {remaining} more question(s) to submit.")
else:
    score = st.session_state.quiz_score
    total = len(questions)
    pct   = int(score / total * 100)
    st.markdown(f"## Final Score: {score}/{total} ({pct}%)")
    if pct == 100:
        st.balloons()
        st.success("Perfect score! You ARE the MCU.")
    elif pct >= 80:
        st.success("Excellent knowledge!")
    elif pct >= 60:
        st.info("Good job — keep watching!")
    else:
        st.warning("Time to re-watch the films!")

    # Show the knowledge base chunks the LLM used to generate ALL questions
    if st.session_state.quiz_chunks:
        with st.expander(f"Knowledge base context used to generate these questions ({len(st.session_state.quiz_chunks)} chunks)"):
            st.caption(
                "All questions were generated from this combined context — "
                "the LLM authored every question holistically, not one chunk per question."
            )
            for j, chunk in enumerate(st.session_state.quiz_chunks, 1):
                meta = chunk.get("metadata", {})
                src  = meta.get("title") or meta.get("filename") or meta.get("source", "Knowledge Base")
                rs   = chunk.get("rerank_score", chunk.get("score", 0))
                st.markdown(f"**Chunk {j}** — `{src}` (relevancy: {rs:.3f})")
                st.markdown(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
                st.divider()
