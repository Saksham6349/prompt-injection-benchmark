"""Google Gemini provider – wraps the google-generativeai SDK."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from src.types import ToolCall

from .base import BaseProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """LLM provider backed by the Google Gemini API."""

    provider_name = "google"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        api_key_env: str = "GOOGLE_API_KEY",
        max_retries: int = 5,
    ):
        try:
            import google.generativeai as genai  # noqa: F811
        except ImportError as exc:
            raise ProviderError(
                "google", "google-generativeai package is not installed"
            ) from exc

        from dotenv import load_dotenv

        load_dotenv()

        resolved_key = (
            api_key
            or os.environ.get(api_key_env, "")
            or os.environ.get("GEMINI_API_KEY", "")
        )
        if not resolved_key:
            raise ProviderError(
                "google", f"No API key: set {api_key_env} or pass api_key"
            )

        genai.configure(api_key=resolved_key)
        self._genai = genai
        self._model = model
        self._model_name = model
        self._max_retries = max_retries

    # ── public ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Call the Gemini GenerativeModel API with optional function calling."""
        # Build Gemini tool declarations
        gemini_tools = self._build_tools(tools) if tools else None

        # Separate system instruction and build content list
        system_instruction = None
        contents: list[Any] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            if role == "system":
                system_instruction = text
            elif role == "assistant":
                parts: list[dict] = []
                if text:
                    parts.append({"text": str(text)})
                for tc in m.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {}
                    else:
                        args = raw_args
                    parts.append({"function_call": {"name": fn_name, "args": args}})
                if not parts:
                    parts = [{"text": "Executing tool call."}]
                contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                # Feed tool results back as function responses
                name = m.get("name", "unknown")
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": name,
                                    "response": {"result": str(text) if text is not None else ""},
                                }
                            }
                        ],
                    }
                )
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        model_kwargs: dict[str, Any] = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            tools=gemini_tools,
            **model_kwargs,
        )

        gen_config = self._genai.types.GenerationConfig(temperature=temperature)

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = model.generate_content(contents, generation_config=gen_config)
                return self._parse(resp)
            except Exception as exc:
                last_exc = exc
                wait = min(2**attempt, 60)
                logger.warning(
                    "Google attempt %d/%d failed (%s), retrying in %ds",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise ProviderError(
            "google", f"All {self._max_retries} retries failed", last_exc
        )

    # ── private ─────────────────────────────────────────────────────

    def _build_tools(self, openai_tools: list[dict]) -> list[Any]:
        """Convert OpenAI function schemas to Gemini function declarations."""
        declarations: list[dict] = []
        for t in openai_tools:
            func = t.get("function", t)
            params = func.get("parameters", {})
            declarations.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": params,
                }
            )
        # Wrap in the protos-style tool object the SDK expects
        return [{"function_declarations": declarations}]

    @staticmethod
    def _parse(resp: Any) -> ProviderResponse:
        """Parse a Gemini GenerateContentResponse into ProviderResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        try:
            candidate = resp.candidates[0]
        except (IndexError, AttributeError):
            return ProviderResponse(content=None, raw_response=resp)

        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCall(name=fc.name, arguments=args))
            elif hasattr(part, "text") and part.text:
                content_parts.append(part.text)

        content = "\n".join(content_parts) if content_parts else None

        usage: dict[str, int] = {}
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            usage = {
                "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(
                    resp.usage_metadata, "candidates_token_count", 0
                ),
            }

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=resp,
        )
