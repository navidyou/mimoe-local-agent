"""Tests for the Agent reasoning loop using a mock LLM.

No running mimOE node required — we substitute a fake LLM whose chat_json()
returns pre-scripted responses so we can exercise every branch of Agent.run().
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.core import Agent
from agent.llm import LocalLLM, Message


def _make_llm(responses: list[dict[str, Any]]) -> LocalLLM:
    """Return a mock LocalLLM that yields pre-scripted chat_json responses."""
    llm = MagicMock(spec=LocalLLM)
    response_iter: Iterator[dict[str, Any]] = iter(responses)
    llm.chat_json.side_effect = lambda *_args, **_kwargs: next(response_iter)
    llm.chat.return_value = "Best effort synthesis answer."
    return llm  # type: ignore[return-value]


def test_agent_direct_answer():
    llm = _make_llm([{"action": "final", "answer": "Paris"}])
    result = Agent(llm).run("What is the capital of France?")
    assert result.answer == "Paris"
    assert len(result.steps) == 1
    assert result.steps[0].kind == "final"


def test_agent_tool_call_then_answer():
    llm = _make_llm([
        {"action": "tool", "tool": "calculator", "input": "8 * 7"},
        {"action": "final", "answer": "8 * 7 is 56."},
    ])
    result = Agent(llm).run("what is 8 * 7?")
    assert result.answer == "8 * 7 is 56."
    assert len(result.steps) == 2
    assert result.steps[0].kind == "tool"
    assert result.steps[0].observation == "56"


def test_agent_unknown_tool_returns_observation():
    llm = _make_llm([
        {"action": "tool", "tool": "nonexistent_tool", "input": "arg"},
        {"action": "final", "answer": "I couldn't do that."},
    ])
    result = Agent(llm).run("use a fake tool")
    assert result.steps[0].observation == "No such tool: 'nonexistent_tool'"
    assert result.answer == "I couldn't do that."


def test_agent_malformed_json_with_no_prior_observation():
    llm = _make_llm([{"_raw": "Sorry I don't understand.\n\nExtra paragraph."}])
    result = Agent(llm).run("??")
    assert result.answer == "Sorry I don't understand."


def test_agent_malformed_json_after_tool_result():
    llm = _make_llm([
        {"action": "tool", "tool": "current_time", "input": ""},
        {"_raw": "The current time is... uh... dunno"},
    ])
    result = Agent(llm).run("what time is it?")
    assert result.answer.startswith("Result:")


def test_agent_loop_exhaustion_triggers_synthesis():
    """When every step is a tool call and the budget runs out, chat() synthesises."""
    tool_call = {"action": "tool", "tool": "calculator", "input": "1 + 1"}
    llm = _make_llm([tool_call] * 10)
    llm.chat.return_value = "The answer is 2."
    result = Agent(llm).run("what is 1+1?")
    assert result.answer == "The answer is 2."


def test_agent_current_time_tool():
    llm = _make_llm([
        {"action": "tool", "tool": "current_time", "input": ""},
        {"action": "final", "answer": "It is some time."},
    ])
    result = Agent(llm).run("what time is it?")
    assert result.steps[0].kind == "tool"
    assert result.steps[0].observation != ""


def test_agent_text_stats_tool():
    llm = _make_llm([
        {"action": "tool", "tool": "text_stats", "input": "hello world"},
        {"action": "final", "answer": "2 words and 11 chars."},
    ])
    result = Agent(llm).run("count hello world")
    assert "words" in result.steps[0].observation


def test_result_has_query_id_in_logs(caplog: pytest.LogCaptureFixture):
    import logging
    llm = _make_llm([{"action": "final", "answer": "42"}])
    with caplog.at_level(logging.INFO, logger="agent.core"):
        Agent(llm).run("test")
    query_ids = {r.query_id for r in caplog.records if hasattr(r, "query_id")}  # type: ignore[attr-defined]
    assert len(query_ids) == 1
