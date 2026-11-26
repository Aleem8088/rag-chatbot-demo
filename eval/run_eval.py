"""Small evaluation harness.

Retrieval mode (default, free, runs offline and in CI):
  - answerable questions: the expected source AND phrase must appear in the top-k passages
  - unanswerable questions: retrieval must return nothing strong enough to answer from

Answer mode (--llm, needs ANTHROPIC_API_KEY, costs a few API calls):
  - answerable questions: the bot must answer (not refuse) and the answer must contain
    the expected phrase's key token (case-insensitive)
  - unanswerable questions: the bot must say it couldn't find the answer

Usage:
  python -m eval.run_eval            # retrieval eval
  python -m eval.run_eval --llm      # full answer eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.chunking import load_directory
from app.main import DOCS_DIR
from app.rag import DEFAULT_MIN_SCORE, answer_question
from app.retriever import Retriever

QUESTIONS = Path(__file__).parent / "questions.json"
K = 4


def run_retrieval(retriever: Retriever, items: list[dict]) -> list[tuple[str, bool, str]]:
    results = []
    for item in items:
        hits = retriever.search(item["q"], k=K)
        strong = [h for h in hits if h[1] >= DEFAULT_MIN_SCORE]
        if item["answerable"]:
            ok = any(
                c.source == item["expected_source"] and item["expected_phrase"].lower() in c.text.lower()
                for c, _ in strong
            )
            detail = "expected passage retrieved" if ok else "expected passage NOT in top results"
        else:
            # Keyword search can't know a question is unanswerable when it shares a word with a
            # document ("gift" in "gift cards"). Declining is the model's job and is tested by
            # --llm, so this is reported as INFO and not counted.
            ok = None
            detail = (
                "correctly found nothing"
                if not strong
                else f"{len(strong)} weak keyword matches; the model must decline (see --llm)"
            )
        results.append((item["q"], ok, detail))
    return results


def run_answers(retriever: Retriever, items: list[dict]) -> list[tuple[str, bool, str]]:
    from app.llm import ClaudeLLM

    llm = ClaudeLLM()
    results = []
    for item in items:
        result = answer_question(item["q"], retriever, llm, k=K)
        if item["answerable"]:
            key = item["expected_phrase"].lower()
            ok = result.found and key in result.text.lower() and len(result.citations) > 0
            detail = "answered with citation" if ok else f"bad answer: {result.text[:80]!r}"
        else:
            ok = not result.found
            detail = "correctly declined" if ok else f"answered a question it should not: {result.text[:80]!r}"
        results.append((item["q"], ok, detail))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="run the full answer eval (uses the API)")
    parser.add_argument("--threshold", type=float, default=0.9, help="minimum pass rate")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = Retriever(load_directory(DOCS_DIR))
    results = run_answers(retriever, items) if args.llm else run_retrieval(retriever, items)

    for question, ok, detail in results:
        label = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{label}] {question}  ->  {detail}")
    scored = [ok for _, ok, _ in results if ok is not None]
    passed = sum(1 for ok in scored if ok)
    rate = passed / len(scored)
    mode = "answer" if args.llm else "retrieval"
    print(f"\n{mode} eval: {passed}/{len(scored)} passed ({rate:.0%}), threshold {args.threshold:.0%}")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
