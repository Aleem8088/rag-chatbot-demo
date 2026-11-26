"""Keyword retrieval with BM25.

BM25 needs no model download and runs anywhere, which keeps the demo easy to run and
easy to test. The Retriever interface is small on purpose: swap in an embeddings or
hybrid retriever for a real project without touching the rest of the app.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.chunking import Chunk

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what",
    "when", "which", "with", "you", "your",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [tokenize(c.text + " " + c.location) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self._bm25:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
