from flask import Flask, request, jsonify
from quiz_utils import (
    load_section,
    retrieve_chunk,
    get_clean_and_question,
    evaluate_answer,
)
import traceback

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "✅ Tutor API is running!"})


@app.route("/ask_question", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(force=True)
        section = data.get("section")
        subtopic = data.get("subtopic")

        index, metadata = load_section(section)
        chunk_text = retrieve_chunk(index, metadata, keyword=subtopic)
        summary, question = get_clean_and_question(chunk_text)

        return jsonify(
            {
                "section": section,
                "subtopic": subtopic,
                "summary": summary,
                "question": question,
                "reference": chunk_text[:300] + "...",
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate_answer", methods=["POST"])
def eval_answer():
    try:
        data = request.get_json(force=True)
        question = data["question"]
        answer = data["answer"]
        ref_text = data.get("reference", "")

        verdict, explanation, example = evaluate_answer(answer, ref_text)

        return jsonify(
            {
                "question": question,
                "student_answer": answer,
                "verdict": verdict,
                "explanation": explanation,
                "example": example,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
