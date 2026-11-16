"""Ollama provider – calls the local Ollama REST API."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from src.types import ToolCall

from .base import BaseProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """LLM provider backed by a local Ollama instance.

    Supports native tool calling for Llama 3.1+ models and falls back to
    text-based function-call parsing for older models.
    """

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout = timeout

    # ── public ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Call the Ollama /api/chat endpoint with optional tool use."""
        import urllib.request
        import urllib.error

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        body = json.dumps(payload).encode()
        url = f"{self._base_url}/api/chat"

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
                return self._parse(data, tools)
            except urllib.error.URLError as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                logger.warning("Ollama attempt %d/%d failed (%s), retrying in %ds",
                               attempt, self._max_retries, exc, wait)
                time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                break

        raise ProviderError("ollama", f"Failed after {self._max_retries} retries", last_exc)

    # ── private ─────────────────────────────────────────────────────

    def _parse(self, data: dict, tools: list[dict]) -> ProviderResponse:
        """Parse an Ollama /api/chat JSON response."""
        message = data.get("message", {})
        content = message.get("content")
        tool_calls: list[ToolCall] = []

        # Native tool calls returned by Llama 3.1+ / Mistral-nemo etc.
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))
        elif content and tools:
            # Fallback: try to extract function calls from text
            tool_calls = self._parse_tool_calls_from_text(content)

        usage: dict[str, int] = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=data,
        )

    @staticmethod
    def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
        """Best-effort extraction of function calls from free-form text.

        Looks for patterns like:
            <function_call>{"name": "...", "arguments": {...}}</function_call>
        or JSON blocks that look like tool calls.
        """
        calls: list[ToolCall] = []

        # Pattern 1: <function_call> tags
        tag_pattern = re.compile(
            r"<function_call>\s*(\{.*?\})\s*</function_call>",
            re.DOTALL,
        )
        for match in tag_pattern.finditer(text):
            try:
                obj = json.loads(match.group(1))
                calls.append(ToolCall(
                    name=obj.get("name", ""),
                    arguments=obj.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                continue

        # Pattern 2: JSON with "name" and "arguments" keys
        if not calls:
            json_pattern = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\"arguments\"\s*:\s*\{[^}]*\}[^}]*\}", re.DOTALL)
            for match in json_pattern.finditer(text):
                try:
                    obj = json.loads(match.group(0))
                    if "name" in obj and "arguments" in obj:
                        calls.append(ToolCall(
                            name=obj["name"],
                            arguments=obj["arguments"],
                        ))
                except json.JSONDecodeError:
                    continue

        return calls
