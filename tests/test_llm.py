"""Tests for LocalLLM using respx to mock the HTTP layer.

No running mimOE node required — these run in CI.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent.llm import LocalLLM, _extract_json


def _chat_response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "smollm-360m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _models_response(model_ids: list[str]) -> dict:
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": 0, "owned_by": "local"} for mid in model_ids],
    }


BASE = "http://localhost:8083/mimik-ai/openai/v1"


@respx.mock
def test_health_check_success():
    respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(200, json=_models_response(["smollm-360m", "qwen3-4b"]))
    )
    llm = LocalLLM()
    models = llm.health_check()
    assert models == ["smollm-360m", "qwen3-4b"]


@respx.mock
def test_health_check_failure():
    respx.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("refused"))
    llm = LocalLLM()
    with pytest.raises(Exception):
        llm.health_check()


@respx.mock
def test_chat_returns_string():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("Hello there!"))
    )
    llm = LocalLLM()
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result == "Hello there!"


@respx.mock
def test_chat_json_parses_valid_json():
    payload = {"action": "final", "answer": "42"}
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(json.dumps(payload)))
    )
    llm = LocalLLM()
    result = llm.chat_json([{"role": "user", "content": "what is 6*7?"}])
    assert result == payload


@respx.mock
def test_chat_json_strips_think_blocks():
    content = "<think>some reasoning here</think>\n{\"action\": \"final\", \"answer\": \"yes\"}"
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(content))
    )
    llm = LocalLLM()
    result = llm.chat_json([{"role": "user", "content": "test"}])
    assert result == {"action": "final", "answer": "yes"}


@respx.mock
def test_chat_json_extracts_from_code_fence():
    content = "```json\n{\"action\": \"final\", \"answer\": \"blue\"}\n```"
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(content))
    )
    llm = LocalLLM()
    result = llm.chat_json([{"role": "user", "content": "test"}])
    assert result == {"action": "final", "answer": "blue"}


@respx.mock
def test_chat_json_fallback_raw():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("I don't know what JSON is"))
    )
    llm = LocalLLM()
    result = llm.chat_json([{"role": "user", "content": "test"}])
    assert "_raw" in result


# --- unit tests for _extract_json (no HTTP) ---

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_think():
    assert _extract_json("<think>reasoning</think>\n{\"x\": 2}") == {"x": 2}


def test_extract_json_unterminated_think():
    assert _extract_json("<think>reasoning that never ends") == {"_raw": ""}


def test_extract_json_brace_fallback():
    assert _extract_json('prefix {"key": "val"} suffix') == {"key": "val"}


def test_extract_json_no_json():
    result = _extract_json("plain text")
    assert "_raw" in result
