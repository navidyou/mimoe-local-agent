# mimOE Local Agent

[![CI](https://github.com/navidyou/mimoe-local-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/navidyou/mimoe-local-agent/actions/workflows/ci.yml)

A small AI agent that runs **entirely on-device** against a local
[mimOE](https://developer.mimik.com/) inference node — no cloud API, no data
leaving the machine. Built with the **BYO-Framework** approach: mimOE exposes an
OpenAI-compatible endpoint, so the standard OpenAI SDK is pointed at the local
node and used unchanged.

```
you > what is 12 * (3 + 4)?
  [1] tool   -> calculator('12 * (3 + 4)')
      result -> 84
  [2] answer -> 12 * (3 + 4) is 84.

12 * (3 + 4) is 84.
```

## How it works

```
            ┌──────────────────────────────────────────────┐
  user      │                  Agent loop                  │
  query ───►│  reason ─► act (tool) ─► observe ─► repeat    │
            │            │                    ▲             │
            └────────────┼────────────────────┼─────────────┘
                         ▼                    │
                ┌─────────────────┐   ┌────────────────┐
                │  LocalLLM        │   │  Tools         │
                │  (OpenAI SDK ──► │   │  calculator    │
                │   mimOE @ :8083) │   │  current_time  │
                └─────────────────┘   │  text_stats    │
                         │            └────────────────┘
                         ▼
                ┌─────────────────────────────────────────┐
                │ mimOE node — OpenAI-compatible inference │
                │ /mimik-ai/openai/v1  (qwen3-4b)         │
                └─────────────────────────────────────────┘
```

Each turn the model returns one JSON decision — either call a tool or give a
final answer. Tool results are fed back, and the loop continues (bounded by
`MIMOE_MAX_STEPS`) until the model answers or the budget is spent.

## Quick start

**Prerequisites:** Python ≥ 3.11 and mimOE Studio running with a model loaded.

```bash
# 1. Install (creates a mimoe-agent entry point)
pip install -e .[dev]
cp .env.example .env        # adjust if your API button shows a different path

# 2. Run
python -m agent.main -q "what time is it?" --trace
python -m agent.main                         # interactive REPL with history
python -m agent.main --version

# Or via the entry point after pip install -e .
mimoe-agent -q "what is 6 * 7?"
```

If mimOE isn't reachable the agent says so immediately and lists the models it
*can* see, rather than hanging.

## Development

```bash
# Install dev dependencies
pip install -e .[dev]

make lint        # ruff check
make typecheck   # mypy --strict
make test        # pytest (no coverage gate)
make test-cov    # pytest with 60% coverage gate
make audit       # pip-audit vulnerability scan
make clean       # remove __pycache__, .mypy_cache, etc.
```

## Logging & tracing

The agent emits structured JSON logs to stderr. Set `LOG_LEVEL=DEBUG` (or pass
`--trace`) to see per-step LLM request/response details and timing:

```bash
LOG_LEVEL=DEBUG python -m agent.main -q "what is 8 * 7?"
python -m agent.main --trace -q "what is 8 * 7?"
```

Each log line includes a `query_id` field for correlating all events belonging
to a single query. Connect any JSON log aggregator (Loki, CloudWatch, etc.) to
stderr for production observability.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MIMOE_BASE_URL` | `http://localhost:8083/mimik-ai/openai/v1` | mimOE OpenAI-compatible endpoint |
| `MIMOE_API_KEY` | `1234` | Bearer token (change for any real deployment) |
| `MIMOE_MODEL` | `qwen3-4b` | Model ID currently loaded in mimOE |
| `MIMOE_TEMPERATURE` | `0.1` | Decoding temperature |
| `MIMOE_MAX_TOKENS` | `512` | Max output tokens per inference |
| `MIMOE_DECISION_MAX_TOKENS` | `1024` | Token budget for the routing decision step |
| `MIMOE_NO_THINK` | `true` | Strip `<think>` blocks from reasoning models (Qwen3 etc.) |
| `MIMOE_MAX_STEPS` | `4` | Hard cap on agent loop iterations |
| `MIMOE_CONNECT_TIMEOUT` | `10` | TCP connect timeout (s) |
| `MIMOE_READ_TIMEOUT` | `60` | Inference read timeout (s) |
| `MIMOE_RETRY_ATTEMPTS` | `3` | Retries on connection/timeout errors |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG` for full traces) |

## Adding a tool

1. Write a plain function `my_tool(arg: str) -> str` in `agent/tools.py`
2. Append `Tool("my_tool", "One-line description.", my_tool)` to the tuple in `REGISTRY`
3. Add a test in `tests/test_tools.py` — tools are the deterministic layer, they must be exact

## Design decisions

**BYO-Framework via the OpenAI SDK, not a heavyweight agent framework.**
mimOE's inference API is OpenAI-compatible, so the simplest, most transparent
integration is the official OpenAI client with `base_url` swapped to the local
node. The whole local-vs-cloud distinction collapses to one config line
(`agent/llm.py`).

**A JSON-routing loop instead of native function-calling.** The models that run
comfortably on-device (SmolLM-360M, Qwen3-4B and similar) do not reliably emit
well-formed OpenAI `tool_calls`. Asking for **one explicit JSON decision per
step**, parsed defensively, is dramatically more robust on this model class.

**The model routes; the tools compute.** Small on-device models are weak at
arithmetic and have no clock, but they are fine at recognising intent. Exactness
lives in plain Python (`agent/tools.py`); the model only does language
understanding.

**Graceful degradation everywhere.** Malformed JSON falls back to treating the
model's text as the answer; unknown tool names return an observation; the loop is
hard-capped; `calculator` uses an AST walker (no `eval`).

**Reasoning model support.** Qwen3 emits `<think>...</think>` blocks before
answering. The agent strips these automatically so the routing parser always sees
clean JSON, regardless of whether the model honours the `/no_think` soft switch.

## Project layout

```
agent/
  config.py         environment-driven settings with startup validation
  llm.py            OpenAI SDK → mimOE (BYO-Framework seam), retry logic
  tools.py          deterministic tools + registry
  core.py           reason → act → observe loop, structured logging + query IDs
  main.py           CLI (one-shot, REPL, --trace / --version)
  logging_config.py JSON formatter, setup_logging()
tests/
  test_tools.py     exactness tests (no model required)
  test_llm.py       mock HTTP tests for LocalLLM
  test_core.py      mock LLM tests for Agent loop branches
  test_logging.py   JSON formatter tests
```

## Limitations & next steps

- **Single-step tool use dominates** on smaller models; richer multi-hop chains
  work better with a larger loaded model (configurable via `MIMOE_MODEL`, no code change).
- **No streaming** in the CLI — easy to add via `stream=True`.
- **Tools are illustrative.** The registry pattern makes adding real tools
  (retrieval, HTTP fetch, local file access) a few lines each.
