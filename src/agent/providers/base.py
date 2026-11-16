"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.types import ToolCall


@dataclass
class ProviderResponse:
    """Standard response from any LLM provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    })
    raw_response: Any = None


class BaseProvider(ABC):
    """Abstract base class that every LLM provider must implement."""

    provider_name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Send a chat request and return a structured response.

        Args:
            messages: Conversation history in OpenAI-style message format.
            tools: Tool definitions in OpenAI function-calling schema.
            temperature: Sampling temperature.

        Returns:
            A ProviderResponse containing content, tool calls, and usage.

        Raises:
            ProviderError: On API or network failures.
        """
        ...


class ProviderError(Exception):
    """Raised when a provider encounters an unrecoverable error."""

    def __init__(self, provider: str, message: str, cause: Exception | None = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}")
