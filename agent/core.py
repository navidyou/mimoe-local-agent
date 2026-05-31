"""The agent: a bounded reason -> act -> observe loop over the local model.

Why a JSON-routing loop instead of OpenAI native function-calling?
  mimOE serves small llama.cpp models (e.g. SmolLM2-360M). These do not reliably
  emit well-formed `tool_calls`, and native function-calling support varies by
  backend. A single, explicitly-prompted JSON decision per step -- parsed
  defensively, with graceful fallback to a plain answer -- is far more robust on
  this class of model while still being a genuine agent (it reasons, picks
  tools, observes results, and decides when to stop).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4

from .config import settings
from .llm import LocalLLM, Message
from .tools import REGISTRY, tool_catalog

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You answer questions, using a tool when one helps.

Tools:
{tool_catalog()}

Reply with EXACTLY ONE JSON object and no other text.
  call a tool: {{"action": "tool", "tool": "<name>", "input": "<arg>"}}
  answer:      {{"action": "final", "answer": "<answer>"}}"""


# Few-shot exemplars shown as prior turns. A 360M model follows the JSON contract
# far more reliably when it has *seen* the format than when it is only told the
# rules -- this is the single highest-leverage change for small-model reliability.
FEW_SHOT: list[dict[str, str]] = [
    {"role": "user", "content": "what time is it?"},
    {"role": "assistant", "content": '{"action": "tool", "tool": "current_time", "input": ""}'},
    {"role": "user", "content": "Tool result: Monday, 02 June 2025, 09:00:00\nNow give the final answer as JSON."},
    {"role": "assistant", "content": '{"action": "final", "answer": "It is 09:00 on Monday, 02 June 2025."}'},
    {"role": "user", "content": "who wrote Hamlet?"},
    {"role": "assistant", "content": '{"action": "final", "answer": "Hamlet was written by William Shakespeare."}'},
    {"role": "user", "content": "what is 8 * 7?"},
    {"role": "assistant", "content": '{"action": "tool", "tool": "calculator", "input": "8 * 7"}'},
]


@dataclass
class Step:
    """One node in the agent's trace, kept for transparency/debugging."""
    kind: str          # "tool" or "final"
    detail: str        # tool call summary or the final answer
    observation: str = ""


@dataclass
class Result:
    answer: str
    steps: list[Step] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LocalLLM | None = None) -> None:
        self.llm = llm or LocalLLM()

    def run(self, query: str) -> Result:
        query_id = str(uuid4())
        t_start = time.monotonic()
        logger.info("Query started", extra={"query_id": query_id, "query": query})

        system = SYSTEM_PROMPT + (" /no_think" if settings.no_think else "")
        messages: list[Message] = [
            {"role": "system", "content": system},
            *FEW_SHOT,
            {"role": "user", "content": query},
        ]
        steps: list[Step] = []
        last_observation: str | None = None

        for step_num in range(settings.max_steps):
            t_step = time.monotonic()
            decision = self.llm.chat_json(messages)
            step_ms = round((time.monotonic() - t_step) * 1000)

            # Defensive fallback: model didn't return usable JSON. If we already
            # have a tool result, return that cleanly instead of dumping the
            # model's free-form ramble (small models tend to wander on the
            # synthesis turn). Otherwise return its text, trimmed to one block.
            if "_raw" in decision:
                if last_observation is not None:
                    answer = f"Result: {last_observation}"
                else:
                    answer = decision["_raw"].strip().split("\n\n")[0].strip()
                steps.append(Step(kind="final", detail=answer))
                logger.debug(
                    "Step: raw fallback",
                    extra={"query_id": query_id, "step": step_num + 1, "latency_ms": step_ms},
                )
                self._log_done(query_id, answer, steps, t_start)
                return Result(answer=answer, steps=steps)

            action = decision.get("action")

            if action == "tool":
                name = decision.get("tool", "")
                tool_input = str(decision.get("input", ""))
                tool = REGISTRY.get(name)
                observation = (
                    tool.func(tool_input) if tool else f"No such tool: {name!r}"
                )
                last_observation = observation
                steps.append(
                    Step(kind="tool", detail=f"{name}({tool_input!r})", observation=observation)
                )
                logger.debug(
                    "Step: tool call",
                    extra={
                        "query_id": query_id,
                        "step": step_num + 1,
                        "tool": name,
                        "input": tool_input,
                        "observation": observation,
                        "latency_ms": step_ms,
                    },
                )
                # Feed the model its own action and the observation, then loop.
                messages.append({"role": "assistant", "content": str(decision)})
                messages.append(
                    {"role": "user", "content": f"Tool result: {observation}\n"
                                                 f"Now give the final answer as JSON."}
                )
                continue

            # action == "final" (or anything else we treat as terminal)
            answer = str(decision.get("answer", "")).strip() or "(no answer)"
            steps.append(Step(kind="final", detail=answer))
            logger.debug(
                "Step: final answer",
                extra={"query_id": query_id, "step": step_num + 1, "latency_ms": step_ms},
            )
            self._log_done(query_id, answer, steps, t_start)
            return Result(answer=answer, steps=steps)

        # Loop budget exhausted: synthesise a best-effort answer from context.
        fallback = self.llm.chat(messages + [
            {"role": "user", "content": "Give your best final answer now in plain text."}
        ])
        steps.append(Step(kind="final", detail=fallback))
        logger.warning(
            "Loop budget exhausted, using synthesis fallback",
            extra={"query_id": query_id, "max_steps": settings.max_steps},
        )
        self._log_done(query_id, fallback, steps, t_start)
        return Result(answer=fallback, steps=steps)

    @staticmethod
    def _log_done(query_id: str, answer: str, steps: list[Step], t_start: float) -> None:
        logger.info(
            "Query completed",
            extra={
                "query_id": query_id,
                "total_steps": len(steps),
                "total_ms": round((time.monotonic() - t_start) * 1000),
                "answer_preview": answer[:120],
            },
        )
