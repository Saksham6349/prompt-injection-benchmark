"""Main agent harness – the core loop that drives tool-using LLM agents.

The harness orchestrates the cycle of:
  user task → LLM → tool calls → tool execution → (defense check) → LLM → …
until the LLM returns a final text response or max_turns is reached.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.types import DefenseVerdict, ToolCall, ToolDefinition, ToolOutput

from .cache import ResponseCache
from .prompts import build_system_prompt
from .providers.base import BaseProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    """Complete record of a single agent run."""

    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    defense_verdicts: list[DefenseVerdict] = field(default_factory=list)
    final_response: str | None = None
    total_latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    })
    turns: int = 0
    error: str | None = None


class AgentHarness:
    """Core agent loop that connects an LLM provider to a tool executor.

    Args:
        provider: The LLM provider to use (OpenAI, Anthropic, etc.)
        tool_executor: Callable that takes a ToolCall and returns a ToolOutput.
        tools: List of tool definitions the agent can use.
        defense_pipeline: Optional callable that takes a ToolCall and ToolOutput
            and returns a DefenseVerdict. None means no defense.
        config: Optional configuration dict (keys: max_turns, temperature, cache).
        cache: Optional ResponseCache instance.
    """

    def __init__(
        self,
        provider: BaseProvider,
        tool_executor: Callable[[ToolCall], ToolOutput],
        tools: list[ToolDefinition] | None = None,
        defense_pipeline: Callable[[ToolCall, ToolOutput], DefenseVerdict] | None = None,
        config: dict[str, Any] | None = None,
        cache: ResponseCache | None = None,
    ):
        self._provider = provider
        self._tool_executor = tool_executor
        self._tools = tools or []
        self._defense_pipeline = defense_pipeline
        self._config = config or {}
        self._cache = cache

        # Pre-compute tool schemas once
        self._tool_schemas = [t.to_openai_schema() for t in self._tools]

    # ── public ──────────────────────────────────────────────────────

    def run(self, user_task: str, max_turns: int | None = None) -> AgentRun:
        """Execute the full agent loop for a given user task.

        Args:
            user_task: The user's natural-language request.
            max_turns: Maximum number of LLM calls (default from config or 10).

        Returns:
            An AgentRun containing the complete trace.
        """
        max_turns = max_turns or self._config.get("max_turns", 10)
        temperature = self._config.get("temperature", 0.0)

        run = AgentRun()
        start = time.perf_counter()

        # Build initial messages
        system_prompt = build_system_prompt(self._tools)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_task},
        ]
        run.messages = list(messages)

        response: ProviderResponse | None = None
        try:
            for turn in range(1, max_turns + 1):
                run.turns = turn
                logger.info("Turn %d/%d", turn, max_turns)

                # 1. Call LLM (with optional cache)
                response = self._call_llm(messages, temperature)
                self._accumulate_usage(run, response.usage)

                # 2. If LLM returns tool calls → execute them
                if response.tool_calls:
                    # Build assistant message with tool calls
                    assistant_msg = self._build_assistant_tool_message(response)
                    messages.append(assistant_msg)
                    run.messages.append(assistant_msg)

                    for tc in response.tool_calls:
                        run.tool_calls.append(tc)
                        logger.info("Tool call: %s(%s)", tc.name, json.dumps(tc.arguments, default=str))

                        # Execute tool
                        tool_output = self._execute_tool(tc)

                        # 3. If defense_pipeline is set → check (with user_task context)
                        verdict = None
                        if self._defense_pipeline is not None:
                            verdict = self._defense_pipeline(tc, tool_output, user_task=user_task)
                            run.defense_verdicts.append(verdict)
                            logger.info("Defense verdict: blocked=%s layer=%s",
                                        verdict.is_blocked, verdict.layer)

                        # 4. If defense blocks → substitute result
                        if verdict and verdict.is_blocked:
                            tool_result_content = (
                                f"[BLOCKED BY POLICY] Tool call '{tc.name}' was blocked by the "
                                f"{verdict.layer} defense layer: {verdict.explanation}"
                            )
                        else:
                            tool_result_content = tool_output.content

                        # Append tool result to messages
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": tool_result_content,
                        }
                        messages.append(tool_msg)
                        run.messages.append(tool_msg)

                else:
                    # 6. LLM returned text (no tool calls) → done
                    run.final_response = response.content
                    if response.content:
                        messages.append({"role": "assistant", "content": response.content})
                        run.messages.append({"role": "assistant", "content": response.content})
                    logger.info("Agent finished with text response on turn %d", turn)
                    break
            else:
                # 7. max_turns reached (BUG-05 fix: safely handle unassigned or tool-exhausted response)
                if response is not None and response.content and not response.tool_calls:
                    run.final_response = response.content
                else:
                    run.final_response = (
                        f"Agent reached maximum turns limit ({max_turns}) without completing task."
                    )
                logger.warning("Agent reached max_turns (%d) without final response", max_turns)

        except ProviderError as exc:
            run.error = str(exc)
            logger.error("Provider error during agent run: %s", exc)
        except Exception as exc:
            run.error = str(exc)
            logger.error("Unexpected error during agent run: %s", exc, exc_info=True)

        run.total_latency_ms = (time.perf_counter() - start) * 1000
        return run

    # ── private ─────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict], temperature: float) -> ProviderResponse:
        """Call the LLM provider, optionally using cache."""
        if self._cache is not None:
            cached = self._cache.get(
                self._provider.provider_name,
                getattr(self._provider, "_model", "unknown"),
                messages,
                self._tool_schemas,
            )
            if cached is not None:
                logger.debug("Using cached response")
                return cached

        response = self._provider.chat(messages, self._tool_schemas, temperature)

        if self._cache is not None:
            self._cache.put(
                self._provider.provider_name,
                getattr(self._provider, "_model", "unknown"),
                messages,
                self._tool_schemas,
                response,
            )

        return response

    def _execute_tool(self, tool_call: ToolCall) -> ToolOutput:
        """Execute a single tool call via the tool executor."""
        try:
            output = self._tool_executor(tool_call)
            tool_call.result = output.content
            return output
        except Exception as exc:
            logger.error("Tool execution error for %s: %s", tool_call.name, exc)
            error_output = ToolOutput(
                tool_name=tool_call.name,
                content=f"Error executing tool '{tool_call.name}': {exc}",
            )
            tool_call.result = error_output.content
            return error_output

    @staticmethod
    def _build_assistant_tool_message(response: ProviderResponse) -> dict:
        """Build an assistant message containing tool_calls (OpenAI format)."""
        tool_calls_data = []
        for tc in response.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, default=str),
                },
            })
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
            "tool_calls": tool_calls_data,
        }
        return msg

    @staticmethod
    def _accumulate_usage(run: AgentRun, usage: dict[str, int]) -> None:
        """Add usage stats from one turn to the cumulative total."""
        run.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        run.usage["completion_tokens"] += usage.get("completion_tokens", 0)
