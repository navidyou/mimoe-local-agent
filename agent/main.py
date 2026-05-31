"""Command-line entrypoint.

Usage:
    python -m agent.main --query "what is 12 * (3 + 4)?"
    python -m agent.main            # interactive REPL
    python -m agent.main --trace    # show the agent's reason/act trace
"""
from __future__ import annotations

import argparse
import sys

from .config import settings
from .core import Agent, Result
from .llm import LocalLLM


def _print_trace(result: Result) -> None:
    for i, step in enumerate(result.steps, 1):
        if step.kind == "tool":
            print(f"  [{i}] tool   -> {step.detail}")
            print(f"      result -> {step.observation}")
        else:
            print(f"  [{i}] answer -> {step.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="A small agent running on a local mimOE model.")
    parser.add_argument("--query", "-q", help="Run a single query and exit.")
    parser.add_argument("--trace", action="store_true", help="Print the agent's tool-use trace.")
    args = parser.parse_args()

    llm = LocalLLM()

    # Fail fast with a useful message if mimOE isn't reachable.
    try:
        loaded = llm.health_check()
    except Exception as exc:  # noqa: BLE001 - surface any connection problem clearly
        print(f"Could not reach mimOE at {settings.base_url}\n  {exc}\n"
              "Is mimOE Studio running with a model loaded?", file=sys.stderr)
        return 1

    print(f"Connected to mimOE at {settings.base_url}")
    print(f"Loaded models: {', '.join(loaded) or '(none)'}")
    if settings.model not in loaded:
        print(f"Configured model {settings.model!r} is not loaded in mimOE.\n"
              f"Set MIMOE_MODEL to one of: {', '.join(loaded) or '(none loaded)'}",
              file=sys.stderr)
        return 1
    print()

    agent = Agent(llm)

    def handle(query: str) -> None:
        result = agent.run(query)
        if args.trace:
            _print_trace(result)
        print(f"\n{result.answer}\n")

    if args.query:
        handle(args.query)
        return 0

    print("Interactive mode. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            query = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            handle(query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
