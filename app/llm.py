"""Thin wrapper around the Anthropic SDK so the rest of the app can be tested with a fake."""

from __future__ import annotations

import os
from typing import Protocol

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
REFUSAL_TEXT = "The model declined to answer this request."


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        # Credentials come from ANTHROPIC_API_KEY (never hardcoded).
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("ANTHROPIC_EFFORT", "low")

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        if response.stop_reason == "refusal":
            return REFUSAL_TEXT
        return "".join(block.text for block in response.content if block.type == "text").strip()
