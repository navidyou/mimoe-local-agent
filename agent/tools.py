"""Tools the agent can call.

Design principle: the *model* does language understanding and routing; the
*tools* do the deterministic work. A 360M model is bad at arithmetic and has no
clock, but it is fine at recognising "the user wants a calculation" and pulling
out the expression. Pushing exactness into plain Python is what makes a small
local model trustworthy.

Each tool is a plain dataclass with a name, a one-line description (shown to the
model), and a callable. Adding a tool is: write a function, append a Tool.
"""
from __future__ import annotations

import ast
import datetime as _dt
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


# --- tool implementations -------------------------------------------------

_ALLOWED_OPS: dict[type[ast.AST], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an arithmetic AST without exposing Python's eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        bin_fn = cast(Callable[[float, float], float], _ALLOWED_OPS[type(node.op)])
        return bin_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        unary_fn = cast(Callable[[float], float], _ALLOWED_OPS[type(node.op)])
        return unary_fn(_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        # Render whole numbers without a trailing .0
        return str(int(result)) if float(result).is_integer() else str(result)
    except Exception:
        return f"Could not evaluate the expression: {expression!r}"


def current_time(_: str) -> str:
    """Return the current local date and time."""
    return _dt.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


def text_stats(text: str) -> str:
    """Return word and character counts for the given text."""
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"


# --- registry --------------------------------------------------------------

REGISTRY: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool("calculator", "Evaluate an arithmetic expression. Arg: the expression string.", calculator),
        Tool("current_time", "Get the current date and time. Arg: ignored.", current_time),
        Tool("text_stats", "Count words and characters in text. Arg: the text.", text_stats),
    )
}


def tool_catalog() -> str:
    """A compact, model-readable list of tools for the system prompt."""
    return "\n".join(f"- {t.name}: {t.description}" for t in REGISTRY.values())
