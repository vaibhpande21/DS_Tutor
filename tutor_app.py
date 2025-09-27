import streamlit as st
import faiss
import json
import random
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# ---- Config ----
EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TIMEOUT = 60
SECTIONS = {
    "Statistics & Probability": "Statistics_Probability",
    "Classical Machine Learning": "Classical Machine Learning",
}

# ---- Setup ----
st.set_page_config(page_title="Data Science Quiz Bot 🎓", layout="wide")
client = OpenAI(api_key=api_key)
embedder = SentenceTransformer(EMBED_MODEL)

# ---- Session State ----
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "question" not in st.session_state:
    st.session_state.question = ""
if "cleaned" not in st.session_state:
    st.session_state.cleaned = ""
if "score" not in st.session_state:
    st.session_state.score = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0, "TOTAL": 0}


# ---- Helper Functions ----
def load_section(section_key):
    sec_folder = Path("vectorstores") / section_key
    index = faiss.read_index(str(sec_folder / "index.faiss"))
    with open(sec_folder / "meta.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def call_openai_chat(system_prompt, user_prompt, temperature=0.2):
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=400,
        timeout=OPENAI_TIMEOUT,
    )
    return resp.choices[0].message.content


def retrieve_chunk(index, metadata, keyword=None, top_k=5):
    if keyword:
        q_emb = embedder.encode([keyword], convert_to_numpy=True).astype("float32")
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        D, I = index.search(q_emb, top_k)
        valid_idxs = [idx for idx in I[0] if idx != -1]
        if not valid_idxs:
            return random.choice(metadata)["text"]
        return metadata[random.choice(valid_idxs)]["text"]
    else:
        return random.choice(metadata)["text"]


def get_clean_and_question(chunk_text: str, max_retries: int = 3):
    """
    Multi-stage refinement:
    1. Summarize chunk into 1–2 sentences.
    2. Generate a self-contained exam-style question from the summary.
    3. Filter out context-heavy questions and regenerate if needed.
    """

    # ---- Step 1: Summarize the concept ----
    system = "You are a helpful Data Science tutor. Return only clean English text."
    user = f"""
    Here is a messy lecture excerpt:

    ---TEXT---
    {chunk_text}
    ---END---

    Task: Summarize the *core concept* in 1–2 sentences.
    Ignore references to figures, tables, lecture excerpt, or examples unless explicitly explained in words.
    """
    summary = call_openai_chat(system, user).strip()

    # ---- Step 2 + 3: Generate question from summary, with filtering ----
    forbidden = re.compile(r"(figure|table|diagram|above|this example)", re.IGNORECASE)

    for attempt in range(max_retries):
        q_system = "You are a Data Science examiner. Return only a valid question."
        q_user = f"""
        Concept summary:
        {summary}

        Task: Generate ONE exam-style conceptual question.
        - The question must be standalone and answerable by someone studying the topic.
        - Do NOT reference figures, tables, diagrams,lecture excerpt, or context-specific phrases.
        """
        question = call_openai_chat(q_system, q_user).strip()

        if not forbidden.search(question):  # passes filter
            return summary, question

        print("⚠️ Discarded bad question, regenerating...")

    # Fallback if all attempts fail
    return summary, "Could not generate a clean question."


def evaluate_answer(user_answer: str, cleaned_context: str):
    """
    Single-call grader tuned to:
      - Return verdict (CORRECT / PARTIAL / INCORRECT)
      - Provide an educational explanation (up to 5 lines)
      - Provide an 'example' field (up to 5 lines)
         * If verdict == INCORRECT: example MUST contain the correct answer / short worked solution
         * Otherwise: example should be a concise illustrative example reinforcing the concept
    Returns: (verdict, explanation, example)
    """
    system = "You are an expert data-science tutor and grader. Be clear, factual, and helpful. Return ONLY valid JSON."
    user = f"""
Context (cleaned lecture excerpt):
{cleaned_context}

Student Answer:
{user_answer}

Instructions:
1) Classify the student's answer as one of: CORRECT, PARTIAL, or INCORRECT (field: verdict).
2) Provide an educational explanation (field: explanation). Up to 5 short lines (but fewer if sufficient).
   - Be clear and helpful.
   - Focus on the core idea tested by the question.
3) Provide content in the field 'example' (up to 5 short lines):
   - If verdict is INCORRECT: write the **correct answer to the original question**. Start with "Correct answer:" so it’s obvious.
   - If verdict is PARTIAL: give a short model answer that fills in the missing details.
   - If verdict is CORRECT: give a short reinforcing example, use-case, or alternative phrasing of the correct concept.
4) The 'example' must not contradict the verdict or explanation.
5) Do not reference figures, tables, or "the above text". Keep it standalone and self-contained.

Return ONLY a JSON object:
{{ "verdict": "...", "explanation": "...", "example": "..." }}
"""
    # single API call
    out = call_openai_chat(system, user, temperature=0.0)

    # try to parse model output as JSON
    try:
        parsed = json.loads(out)
        verdict = parsed.get("verdict", "").upper()
        explanation = parsed.get("explanation", "").strip()
        example = parsed.get("example", "").strip()
    except Exception:
        # fallback: if model didn't return strict JSON, return a conservative response
        # Put the raw output into explanation so the student still learns something.
        verdict = "UNKNOWN"
        explanation = out.strip()[:2000]
        example = ""

    # Lightweight post-check / safety net (no extra API call)
    # If verdict is INCORRECT but the example is empty or too short, use the explanation as the example (so student gets the answer).
    if verdict == "INCORRECT":
        if not example or len(example) < 40:
            # if explanation is long enough, promote it to example so the user receives the correct answer
            if explanation and len(explanation) >= 40:
                example = explanation
            else:
                # As last resort, set example to a short generic corrected response
                example = "Correct answer: (see explanation above)."

    # ensure returned strings are not excessively long (trim to ~5 lines)
    def trim_to_lines(text, max_lines=5):
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[:max_lines])

    explanation = trim_to_lines(explanation, 5)
    example = trim_to_lines(example, 5)

    return verdict, explanation, example


# ---- Sidebar ----
st.sidebar.title("⚙️ Quiz Settings")
section = st.sidebar.selectbox("Choose a Section", list(SECTIONS.keys()))
subtopic = st.sidebar.text_input("Subtopic (optional)", "")

if st.sidebar.button("Load Section"):
    st.session_state.index, st.session_state.metadata = load_section(SECTIONS[section])
    st.session_state.score = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0, "TOTAL": 0}
    st.success(f"Loaded section: {section}")

# ---- Main Panel ----
st.title("📘 Data Science Quiz Bot")

if st.session_state.index:
    if st.button("Next Question"):
        raw_text = retrieve_chunk(
            st.session_state.index,
            st.session_state.metadata,
            keyword=subtopic if subtopic else None,
        )
        cleaned, question = get_clean_and_question(raw_text)
        st.session_state.cleaned = cleaned
        st.session_state.question = question

    if st.session_state.question:
        st.subheader(f"🤖 Question:")
        st.write(st.session_state.question)

        user_answer = st.text_area("✍️ Your Answer", "")
        if st.button("Submit Answer"):
            verdict, explanation, example = evaluate_answer(
                user_answer, st.session_state.cleaned
            )
            st.session_state.score["TOTAL"] += 1
            if verdict in st.session_state.score:
                st.session_state.score[verdict] += 1

            st.markdown(f"**Verdict:** {verdict}")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown(f"**Example/Model Answer:** {example}")

    # ---- Score Tracking ----
    st.sidebar.subheader("📊 Score")
    st.sidebar.write(st.session_state.score)
else:
    st.info("👉 Load a section from the sidebar to start the quiz.")
