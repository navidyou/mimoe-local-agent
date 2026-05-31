"""Thin wrapper around the official OpenAI SDK, pointed at the local mimOE node.

This file *is* the "BYO Framework" integration: mimOE serves an OpenAI-compatible
API, so the canonical OpenAI client works unchanged once `base_url` is swapped.
Keeping this isolated means the rest of the agent never knows or cares that
inference is local -- swapping mimOE for any other OpenAI-compatible backend is
a one-line config change.
"""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .config import settings


class LocalLLM:
    """Minimal chat client for the mimOE inference endpoint."""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )

    def health_check(self) -> list[str]:
        """Return the ids of models loaded in mimOE.

        Used at startup so the agent fails with a clear, actionable message
        ("is mimOE running? is a model loaded?") instead of a cryptic timeout
        mid-conversation.
        """
        models = self._client.models.list()
        return [m.id for m in models.data]

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        """Single chat completion -> assistant text."""
        resp = self._client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=overrides.get("temperature", settings.temperature),
            max_tokens=overrides.get("max_tokens", settings.max_tokens),
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Chat completion whose response is parsed as a JSON object.

        We do NOT rely on the server's `response_format=json_object` flag: the
        small llama.cpp-backed models in mimOE don't reliably honour it. Instead
        we prompt for JSON and parse defensively, tolerating code fences and
        leading/trailing prose. If parsing fails, the caller treats the raw text
        as a plain answer -- the agent degrades gracefully rather than crashing.
        """
        raw = self.chat(messages, temperature=0.0, max_tokens=settings.decision_max_tokens)
        return _extract_json(raw)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from model output."""
    # Reasoning models (Qwen3, DeepSeek-R1, etc.) wrap chain-of-thought in
    # <think>...</think>. Strip it so it can't confuse JSON parsing. Also handle
    # an unterminated think block (output truncated mid-reasoning).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to grabbing the outermost {...} span.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Signal "no structured decision" to the caller.
    return {"_raw": text}
