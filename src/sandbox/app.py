"""FastAPI application exposing the sandbox tools over HTTP."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.sandbox.state import SandboxState
from src.sandbox.executor import ToolExecutor

# ── Application & shared state ─────────────────────────────────────────

app = FastAPI(
    title="Prompt-Injection Benchmark — Sandbox",
    description="Mock tool environment for evaluating LLM agent defenses.",
    version="0.1.0",
)

_state = SandboxState()
_executor = ToolExecutor(state=_state)


# ── Middleware: log every tool call ─────────────────────────────────────

@app.middleware("http")
async def log_tool_calls(request: Request, call_next):
    """Log POST requests to ``/tools/*`` in the ``tool_call_log`` table."""
    response = await call_next(request)
    # Logging is handled inside ToolExecutor.execute, so the middleware
    # only needs to pass through.  We keep it here as an extension point
    # (e.g. for timing or request-ID injection).
    return response


# ── Routes ──────────────────────────────────────────────────────────────

@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Request) -> JSONResponse:
    """Unified endpoint that routes to the correct tool function.

    Expects a JSON body with the tool arguments, e.g.::

        POST /tools/read_inbox
        {"user": "alice"}
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}

    result = _executor.execute(tool_name, body)
    return JSONResponse(
        content={
            "tool_name": result.tool_name,
            "content": result.content,
            "channel": result.channel.value if result.channel else None,
            "metadata": result.metadata,
        }
    )


@app.post("/reset")
async def reset_state() -> JSONResponse:
    """Reset the sandbox state to its deterministic seed data."""
    _executor.reset()
    return JSONResponse(content={"status": "ok", "message": "Sandbox state reset to seed data."})


@app.get("/state")
async def get_state() -> JSONResponse:
    """Return a full snapshot of the sandbox database for outcome checking."""
    snapshot = _state.get_state_snapshot()
    return JSONResponse(content=snapshot)


@app.get("/tools")
async def list_tools() -> JSONResponse:
    """List all available tool definitions in OpenAI function-calling format."""
    defs = _executor.get_all_tool_definitions()
    return JSONResponse(content=[d.to_openai_schema() for d in defs])
