"""Flask app: ask questions over documents, upload more documents, get cited answers."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from app.chunking import SUPPORTED_EXTENSIONS, chunk_file_bytes, load_directory
from app.llm import LLM, ClaudeLLM
from app.rag import answer_question
from app.retriever import Retriever

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
MAX_QUESTION_CHARS = 500


def create_app(llm: LLM | None = None, docs_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB per upload

    state = {"chunks": load_directory(docs_dir or DOCS_DIR), "llm": llm}
    state["retriever"] = Retriever(state["chunks"])

    def get_llm() -> LLM:
        if state["llm"] is None:
            state["llm"] = ClaudeLLM()  # needs ANTHROPIC_API_KEY
        return state["llm"]

    @app.get("/")
    def index():
        sources = sorted({c.source for c in state["chunks"]})
        return render_template("index.html", sources=sources)

    @app.get("/health")
    def health():
        return jsonify(status="ok", chunks=len(state["chunks"]))

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify(error="Please enter a question."), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify(error=f"Question is too long (max {MAX_QUESTION_CHARS} characters)."), 400
        try:
            result = answer_question(question, state["retriever"], get_llm())
        except Exception as exc:  # surface a clean error, never a stack trace or key
            app.logger.exception("ask failed")
            return jsonify(error=f"The assistant is unavailable ({type(exc).__name__})."), 502
        return jsonify(
            answer=result.text,
            found=result.found,
            citations=[c.__dict__ for c in result.citations],
        )

    @app.post("/upload")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="No file provided."), 400
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            return jsonify(error=f"Unsupported file type. Allowed: {allowed}"), 400
        name = os.path.basename(file.filename)
        try:
            new_chunks = chunk_file_bytes(name, file.read())
        except Exception:
            return jsonify(error="Could not read that file."), 400
        if not new_chunks:
            return jsonify(error="The file contained no readable text."), 400
        state["chunks"] = [c for c in state["chunks"] if c.source != name] + new_chunks
        state["retriever"] = Retriever(state["chunks"])
        return jsonify(source=name, chunks=len(new_chunks), total_chunks=len(state["chunks"]))

    return app


app = create_app()
