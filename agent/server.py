"""HTTP server that exposes the agent as a REST API.

This is the entrypoint for mim/container deployment. The CLI (main.py) remains
available for local development and one-shot queries.

Endpoints
---------
POST /query          Run a query through the agent loop.
GET  /health         Liveness probe — also verifies mimOE connectivity.
GET  /models         List models currently loaded in mimOE.

Environment
-----------
SERVER_HOST          Bind address (default: 0.0.0.0)
SERVER_PORT          Port (default: 3000)

All MIMOE_* and LOG_LEVEL variables from config.py are also respected.

Usage
-----
    # Dev
    uvicorn agent.server:app --reload --port 3000

    # Production / mim container
    python -m agent.server

    # Docker
    docker run -p 3000:3000 -e MIMOE_API_KEY=... mimoe-agent-server
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import settings
from .core import Agent, Result
from .llm import LocalLLM
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state — initialised once at startup, reused across requests.
# ---------------------------------------------------------------------------
_llm: LocalLLM | None = None
_agent: Agent | None = None
_loaded_models: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect to mimOE and fail fast if it's unreachable."""
    global _llm, _agent, _loaded_models

    setup_logging(settings.log_level)
    logger.info("Server starting", extra={"base_url": settings.base_url, "model": settings.model})

    llm = LocalLLM()
    try:
        _loaded_models = llm.health_check()
    except Exception as exc:
        logger.error(
            "Cannot reach mimOE — refusing to start",
            extra={"base_url": settings.base_url, "error": str(exc)},
        )
        raise RuntimeError(
            f"mimOE not reachable at {settings.base_url}. "
            "Start mimOE Studio with a model loaded before running this server."
        ) from exc

    if settings.model not in _loaded_models:
        raise RuntimeError(
            f"Configured model '{settings.model}' is not loaded in mimOE. "
            f"Loaded: {_loaded_models}. Set MIMOE_MODEL to one of those."
        )

    _llm = llm
    _agent = Agent(_llm)
    logger.info("Server ready", extra={"loaded_models": _loaded_models, "model": settings.model})

    yield  # server runs here

    logger.info("Server shutting down")


app = FastAPI(
    title="mimOE Local Agent",
    description="On-device AI agent backed by a local mimOE inference node.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="The question or instruction for the agent.")


class StepResponse(BaseModel):
    kind: str
    detail: str
    observation: str


class QueryResponse(BaseModel):
    answer: str
    steps: list[StepResponse]
    total_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    loaded_models: list[str]
    base_url: str


class ModelsResponse(BaseModel):
    models: list[str]


# ---------------------------------------------------------------------------
# Middleware — add request-level logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    t0 = time.monotonic()
    response = await call_next(request)
    logger.info(
        "HTTP request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        },
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness + readiness probe. Returns 503 if mimOE is not reachable."""
    if _agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not initialised")
    return HealthResponse(
        status="ok",
        model=settings.model,
        loaded_models=_loaded_models,
        base_url=settings.base_url,
    )


@app.get("/models", response_model=ModelsResponse, tags=["ops"])
def list_models() -> ModelsResponse:
    """Return the models currently loaded in mimOE."""
    if _llm is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM not initialised")
    try:
        models = _llm.health_check()
        return ModelsResponse(models=models)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse, tags=["agent"])
def run_query(body: QueryRequest) -> QueryResponse:
    """Run a query through the agent loop and return the answer with trace."""
    if _agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not initialised")

    t0 = time.monotonic()
    try:
        result: Result = _agent.run(body.query)
    except Exception as exc:
        logger.error("Agent run failed", extra={"query": body.query, "error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        steps=[StepResponse(kind=s.kind, detail=s.detail, observation=s.observation) for s in result.steps],
        total_ms=round((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run() -> None:
    """Entry point for `mimoe-agent-server` CLI command and `python -m agent.server`."""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "3000"))
    uvicorn.run("agent.server:app", host=host, port=port, log_config=None)


if __name__ == "__main__":
    run()
