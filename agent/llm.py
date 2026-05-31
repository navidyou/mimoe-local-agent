"""Thin wrapper around the official OpenAI SDK, pointed at the local mimOE node.

This file *is* the "BYO Framework" integration: mimOE serves an OpenAI-compatible
API, so the canonical OpenAI client works unchanged once `base_url` is swapped.
Keeping this isolated means the rest of the agent never knows or cares that
inference is local -- swapping mimOE for any other OpenAI-compatible backend is
a one-line config change.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, cast

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import settings

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LocalLLM:
    """Minimal chat client for the mimOE inference endpoint."""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout,
                read=settings.read_timeout,
                write=settings.connect_timeout,
                pool=settings.connect_timeout,
            ),
        )

    def health_check(self) -> list[str]:
        """Return the ids of models loaded in mimOE.

        Used at startup so the agent fails with a clear, actionable message
        ("is mimOE running? is a model loaded?") instead of a cryptic timeout
        mid-conversation.
        """
        try:
            models = self._client.models.list()
            ids = [m.id for m in models.data]
            logger.info("mimOE health check passed", extra={"loaded_models": ids, "base_url": settings.base_url})
            return ids
        except Exception:
            logger.error("mimOE health check failed", extra={"base_url": settings.base_url}, exc_info=True)
            raise

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(settings.retry_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def chat(self, messages: list[Message], **overrides: Any) -> str:
        """Single chat completion -> assistant text."""
        temperature = overrides.get("temperature", settings.temperature)
        max_tokens = overrides.get("max_tokens", settings.max_tokens)
        logger.debug(
            "LLM request",
            extra={
                "model": settings.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message_count": len(messages),
            },
        )
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=settings.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = round((time.monotonic() - t0) * 1000)
        content = (resp.choices[0].message.content or "").strip()
        logger.debug(
            "LLM response",
            extra={
                "latency_ms": latency_ms,
                "finish_reason": resp.choices[0].finish_reason,
                "output_tokens": resp.usage.completion_tokens if resp.usage else None,
                "response_preview": content[:120],
            },
        )
        return content

    def chat_json(self, messages: list[Message]) -> dict[str, Any]:
        """Chat completion whose response is parsed as a JSON object.

        We do NOT rely on the server's `response_format=json_object` flag: the
        small llama.cpp-backed models in mimOE don't reliably honour it. Instead
        we prompt for JSON and parse defensively, tolerating code fences and
        leading/trailing prose. If parsing fails, the caller treats the raw text
        as a plain answer -- the agent degrades gracefully rather than crashing.
        """
        raw = self.chat(messages, temperature=0.0, max_tokens=settings.decision_max_tokens)
        result = _extract_json(raw)
        if "_raw" in result:
            logger.debug("JSON extraction failed, returning raw text", extra={"raw_preview": raw[:120]})
        return result


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from model output."""
    # Reasoning models (Qwen3, DeepSeek-R1, etc.) wrap chain-of-thought in
    # <think>...</think>. Strip it so it can't confuse JSON parsing. Also handle
    # an unterminated think block (output truncated mid-reasoning).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return cast(dict[str, Any], json.loads(cleaned))
    except json.JSONDecodeError:
        pass
    # Fall back to grabbing the outermost {...} span.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return cast(dict[str, Any], json.loads(cleaned[start : end + 1]))
        except json.JSONDecodeError:
            pass
    # Signal "no structured decision" to the caller.
    return {"_raw": text}
