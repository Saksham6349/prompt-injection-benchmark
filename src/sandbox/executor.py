"""In-process tool executor for fast benchmark runs (no HTTP overhead)."""

from __future__ import annotations

import json
from typing import Any

from src.sandbox.state import SandboxState
from src.sandbox.tools import email, calendar, filestore, web_fetch, bank
from src.types import ToolDefinition, ToolOutput


# ── Dispatch Table ──────────────────────────────────────────────────────

_TOOL_DISPATCH: dict[str, Any] = {
    # Email
    "read_inbox": lambda st, args: email.read_inbox(st, **args),
    "read_email": lambda st, args: email.read_email(st, **args),
    "send_email": lambda st, args: email.send_email(st, **args),
    "search_emails": lambda st, args: email.search_emails(st, **args),
    # Calendar
    "list_events": lambda st, args: calendar.list_events(st, **args),
    "get_event": lambda st, args: calendar.get_event(st, **args),
    "create_event": lambda st, args: calendar.create_event(st, **args),
    "update_event": lambda st, args: calendar.update_event(st, **args),
    "delete_event": lambda st, args: calendar.delete_event(st, **args),
    # File store
    "list_files": lambda st, args: filestore.list_files(st, **args),
    "read_file": lambda st, args: filestore.read_file_tool(st, **args),
    "create_file": lambda st, args: filestore.create_file(st, **args),
    "delete_file": lambda st, args: filestore.delete_file(st, **args),
    "search_files": lambda st, args: filestore.search_files(st, **args),
    # Web
    "fetch_page": lambda st, args: web_fetch.fetch_page(st, **args),
    # Bank
    "check_balance": lambda st, args: bank.check_balance(st, **args),
    "transfer": lambda st, args: bank.transfer(st, **args),
    "transaction_history": lambda st, args: bank.transaction_history(st, **args),
}


class ToolExecutor:
    """Execute sandbox tools in-process without HTTP.

    Provides the same interface as calling the FastAPI endpoints
    but avoids network overhead — useful for tight benchmark loops.
    """

    def __init__(self, state: SandboxState | None = None) -> None:
        self.state = state or SandboxState()

    def __call__(self, tool_call: Any) -> ToolOutput:
        """Allow calling ToolExecutor directly with a ToolCall instance."""
        name = getattr(tool_call, "name", "")
        args = getattr(tool_call, "arguments", {})
        return self.execute(name, args)

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolOutput:
        """Run a tool by name and return its ``ToolOutput``.

        Args:
            tool_name: Registered tool name (e.g. ``"read_inbox"``).
            arguments: Keyword arguments forwarded to the tool function.

        Returns:
            A ``ToolOutput`` with the tool's textual result.

        Raises:
            ValueError: If *tool_name* is not recognised.
        """
        arguments = arguments or {}
        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return ToolOutput(
                tool_name=tool_name,
                content=f"Error: unknown tool '{tool_name}'. Available: {', '.join(sorted(_TOOL_DISPATCH))}",
            )

        try:
            result: ToolOutput = handler(self.state, arguments)
        except Exception as exc:  # noqa: BLE001
            result = ToolOutput(tool_name=tool_name, content=f"Error executing {tool_name}: {exc}")

        # Log the call
        self.state.log_tool_call(
            tool_name,
            json.dumps(arguments, default=str),
            result.content,
        )
        return result

    def get_all_tool_definitions(self) -> list[ToolDefinition]:
        """Aggregate tool definitions from every tool module."""
        defs: list[ToolDefinition] = []
        defs.extend(email.get_tool_definitions())
        defs.extend(calendar.get_tool_definitions())
        defs.extend(filestore.get_tool_definitions())
        defs.extend(web_fetch.get_tool_definitions())
        defs.extend(bank.get_tool_definitions())
        return defs

    def reset(self) -> None:
        """Reset the underlying sandbox state to seed data."""
        self.state.reset()
