"""Command-line entrypoint.

Usage:
    python -m agent.main --query "what is 12 * (3 + 4)?"
    python -m agent.main            # interactive REPL
    python -m agent.main --trace    # show DEBUG logs (LLM traces + per-step timing)
"""
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import sys

from .config import settings
from .core import Agent, Result
from .llm import LocalLLM
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


def _print_trace(result: Result) -> None:
    for i, step in enumerate(result.steps, 1):
        if step.kind == "tool":
            print(f"  [{i}] tool   -> {step.detail}")
            print(f"      result -> {step.observation}")
        else:
            print(f"  [{i}] answer -> {step.detail}")


def main() -> int:
    try:
        _version = importlib.metadata.version("mimoe-local-agent")
    except importlib.metadata.PackageNotFoundError:
        _version = "0.0.0-dev"

    parser = argparse.ArgumentParser(description="A small agent running on a local mimOE model.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {_version}")
    parser.add_argument("--query", "-q", help="Run a single query and exit.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Set log level to DEBUG: prints full LLM request/response traces and per-step timing.",
    )
    args = parser.parse_args()

    log_level = "DEBUG" if args.trace else settings.log_level
    setup_logging(log_level)

    llm = LocalLLM()

    # Fail fast with a useful message if mimOE isn't reachable.
    try:
        loaded = llm.health_check()
    except Exception as exc:
        logger.error(
            "Could not reach mimOE",
            extra={"base_url": settings.base_url, "error": str(exc)},
        )
        print(
            f"Could not reach mimOE at {settings.base_url}\n  {exc}\n"
            "Is mimOE Studio running with a model loaded?",
            file=sys.stderr,
        )
        return 1

    print(f"Connected to mimOE at {settings.base_url}")
    print(f"Loaded models: {', '.join(loaded) or '(none)'}")
    if settings.model not in loaded:
        logger.error(
            "Configured model not loaded",
            extra={"model": settings.model, "loaded": loaded},
        )
        print(
            f"Configured model {settings.model!r} is not loaded in mimOE.\n"
            f"Set MIMOE_MODEL to one of: {', '.join(loaded) or '(none loaded)'}",
            file=sys.stderr,
        )
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
