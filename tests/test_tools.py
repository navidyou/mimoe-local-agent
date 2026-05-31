"""Tests for the deterministic tools.

These need no model and no running mimOE node, so they run anywhere (incl. CI)
and prove the part of the system that must be exact actually is exact.
"""
from agent.tools import REGISTRY, calculator, text_stats


def test_calculator_basic():
    assert calculator("12 * (3 + 4)") == "84"
    assert calculator("2 ** 10") == "1024"
    assert calculator("10 / 4") == "2.5"


def test_calculator_rejects_unsafe_input():
    # No code execution path: arbitrary names/calls are not evaluated.
    assert calculator("__import__('os').system('echo hi')").startswith("Could not evaluate")


def test_text_stats():
    assert text_stats("hello world") == "2 words, 11 characters"


def test_registry_contract():
    for name, tool in REGISTRY.items():
        assert tool.name == name
        assert callable(tool.func)
        assert tool.description
