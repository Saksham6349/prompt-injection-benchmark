"""Anthropic provider – wraps the anthropic SDK."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from src.types import ToolCall

from .base import BaseProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """LLM provider backed by the Anthropic Messages API."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_retries: int = 5,
        max_tokens: int = 4096,
    ):
        try:
            import anthropic  # noqa: F811
        except ImportError as exc:
            raise ProviderError("anthropic", "anthropic package is not installed") from exc

        resolved_key = api_key or os.environ.get(api_key_env, "")
        if not resolved_key:
            raise ProviderError("anthropic", f"No API key: set {api_key_env} or pass api_key")

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    # ── public ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Call the Anthropic Messages API with optional tool use."""
        # Anthropic expects system prompt separately
        system_text = ""
        api_messages: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                system_text += m.get("content", "") + "\n"
            else:
                api_messages.append(m)

        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = [self._convert_tool(t) for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
            "temperature": temperature,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.messages.create(**kwargs)
                return self._parse(resp)
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status and status < 500 and status != 429:
                    break
                wait = min(2 ** attempt, 60)
                logger.warning("Anthropic attempt %d/%d failed (%s), retrying in %ds",
                               attempt, self._max_retries, exc, wait)
                time.sleep(wait)

        raise ProviderError("anthropic", f"All {self._max_retries} retries failed", last_exc)

    # ── private ─────────────────────────────────────────────────────

    @staticmethod
    def _convert_tool(openai_tool: dict) -> dict:
        """Convert an OpenAI function-calling schema to Anthropic tool format."""
        func = openai_tool.get("function", openai_tool)
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        }

    @staticmethod
    def _parse(resp: Any) -> ProviderResponse:
        """Parse an Anthropic Messages response into ProviderResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        content = "\n".join(content_parts) if content_parts else None

        usage: dict[str, int] = {}
        if hasattr(resp, "usage") and resp.usage:
            usage = {
                "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
                "completion_tokens": getattr(resp.usage, "output_tokens", 0),
            }

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=resp,
        )
