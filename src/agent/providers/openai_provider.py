"""OpenAI provider – wraps the openai SDK."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from src.types import ToolCall

from .base import BaseProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """LLM provider backed by the OpenAI Chat Completions API."""

    provider_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 5,
        base_url: str | None = None,
    ):
        try:
            import openai  # noqa: F811
        except ImportError as exc:
            raise ProviderError("openai", "openai package is not installed") from exc

        resolved_key = api_key or os.environ.get(api_key_env, "")
        if not resolved_key:
            raise ProviderError("openai", f"No API key: set {api_key_env} or pass api_key")

        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._model = model
        self._max_retries = max_retries

    # ── public ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Call the OpenAI Chat Completions API with optional tool use."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return self._parse(resp)
            except Exception as exc:
                last_exc = exc
                # Retry on rate-limit (429) or server errors (5xx)
                status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
                if status and status < 500 and status != 429:
                    break
                wait = min(2 ** attempt, 60)
                logger.warning("OpenAI attempt %d/%d failed (%s), retrying in %ds",
                               attempt, self._max_retries, exc, wait)
                time.sleep(wait)

        raise ProviderError("openai", f"All {self._max_retries} retries failed", last_exc)

    # ── private ─────────────────────────────────────────────────────

    def _parse(self, resp: Any) -> ProviderResponse:
        """Parse an OpenAI ChatCompletion response into ProviderResponse."""
        choice = resp.choices[0]
        message = choice.message

        content = message.content
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage: dict[str, int] = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=resp,
        )
