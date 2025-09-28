import streamlit as st
import requests
from pathlib import Path
import os
from dotenv import load_dotenv

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:5001")  # default if not set
VECTOR_DIR = Path("vectorstores")

# -------------------------
# Setup
# -------------------------
st.set_page_config(page_title="Data Science Quiz Bot 🎓", layout="wide")

# ---- Session State ----
if "question" not in st.session_state:
    st.session_state.question = ""
if "reference" not in st.session_state:
    st.session_state.reference = ""
if "score" not in st.session_state:
    st.session_state.score = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0, "TOTAL": 0}

# ---- Sidebar ----
st.sidebar.title("⚙️ Quiz Settings")

# Sections = vectorstore folders
sections = [p.name for p in VECTOR_DIR.iterdir() if p.is_dir()]
section = st.sidebar.selectbox("Choose a Section", sections)
subtopic = st.sidebar.text_input("Subtopic (optional)", "")

# ---- Main Panel ----
st.title("📘 Data Science Quiz Bot")

# ---- Next Question ----
if st.button("Next Question"):
    payload = {"section": section, "subtopic": subtopic}
    try:
        res = requests.post(f"{API_URL}/ask_question", json=payload)
        if res.status_code == 200:
            data = res.json()
            st.session_state.question = data["question"]
            st.session_state.reference = data["reference"]
            st.subheader("🤖 Question:")
            st.write(data["question"])
        else:
            st.error(f"Failed to fetch question (status {res.status_code})")
    except Exception as e:
        st.error(f"Error connecting to API ({API_URL}): {e}")

# ---- Submit Answer ----
if st.session_state.question:
    st.subheader("✍️ Your Answer")
    user_answer = st.text_area("", "")

    if st.button("Submit Answer") and user_answer.strip():
        payload = {
            "section": section,
            "question": st.session_state.question,
            "answer": user_answer,
            "reference": st.session_state.reference,
        }
        try:
            res = requests.post(f"{API_URL}/evaluate_answer", json=payload)
            if res.status_code == 200:
                result = res.json()
                st.session_state.score["TOTAL"] += 1
                if result["verdict"] in st.session_state.score:
                    st.session_state.score[result["verdict"]] += 1

                st.markdown(f"**Verdict:** {result['verdict']}")
                st.markdown(f"**Explanation:** {result['explanation']}")
                st.markdown(f"**Example/Model Answer:** {result['example']}")
            else:
                st.error(f"Failed to evaluate answer (status {res.status_code})")
        except Exception as e:
            st.error(f"Error connecting to API ({API_URL}): {e}")

# ---- Score Tracking ----
st.sidebar.subheader("📊 Score")
st.sidebar.write(st.session_state.score)
