"""Configuration for the mimOE local agent.

All values are read from the environment so the same code runs against a
laptop's local mimOE node, a teammate's node on the mesh, or CI, without edits.
Defaults match a stock mimOE AI Foundation install.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Load a local .env file if present so the documented .env workflow actually
# works. Falls back silently to real environment variables if python-dotenv
# isn't installed or no .env exists.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    # Base URL of the OpenAI-compatible inference endpoint exposed by mimOE.
    #   AI Foundation (mILM) default: http://localhost:8083/mimik-ai/openai/v1
    #   ai-router mim alternative:    http://localhost:8083/mimik-airouter/openai/v1
    # Confirm the exact path via Model View -> API button in mimOE Studio.
    base_url: str = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-ai/openai/v1")

    # Bearer token. Stock mimOE ships with "1234"; override for any real node.
    api_key: str = os.getenv("MIMOE_API_KEY", "1234")

    # Id of a model currently loaded in mimOE. List loaded models with
    #   curl -H "Authorization: Bearer 1234" $MIMOE_BASE_URL/models
    model: str = os.getenv("MIMOE_MODEL", "smollm-360m")

    # Decoding controls. Low temperature keeps the small model's tool-routing
    # decisions stable and repeatable.
    temperature: float = float(os.getenv("MIMOE_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MIMOE_MAX_TOKENS", "512"))

    # Token budget for the structured routing decision. Sized so a reasoning
    # model (e.g. Qwen3) can complete its <think> block AND still emit the JSON
    # after it -- a tight cap truncates mid-thought and starves the decision.
    # The real anti-ramble protection is JSON extraction + graceful fallback
    # (see llm._extract_json), not this number; it only bounds runaway output.
    decision_max_tokens: int = int(os.getenv("MIMOE_DECISION_MAX_TOKENS", "1024"))

    # Reasoning models emit <think>...</think> before answering. mimOE applies a
    # generic chat template, so Qwen3's "/no_think" soft switch is often NOT
    # honoured -- we therefore rely on stripping <think> blocks rather than on
    # disabling thinking. This flag still appends the switch in case a future
    # template supports it; it is harmless when ignored.
    no_think: bool = os.getenv("MIMOE_NO_THINK", "true").lower() == "true"

    # Hard cap on the agent's reason->act loop. Small models can ramble, so we
    # never let the loop run unbounded.
    max_steps: int = int(os.getenv("MIMOE_MAX_STEPS", "4"))

    # Network timeout (seconds) for a single inference call.
    request_timeout: float = float(os.getenv("MIMOE_TIMEOUT", "60"))


settings = Settings()
