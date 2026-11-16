"""Layer 3: Tool-Call Policy Check.

Blocks tool calls that look like they were not requested by the user,
especially for sensitive operations.
"""

from __future__ import annotations

import re
import sys
import os
from typing import Any, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.types import DefenseConfig, DefenseVerdict, ToolCall


# Sensitive tools: actions that can cause harm if triggered by an injection.
DEFAULT_SENSITIVE_TOOLS: set[str] = {
    "send_email",
    "send_message",
    "delete_file",
    "delete_email",
    "transfer",
    "bank_transfer",
    "transfer_money",
    "execute_code",
    "run_command",
    "create_payment",
    "modify_permissions",
    "update_password",
    "post_social_media",
    "create_calendar_event",
    "delete_event",
}

# Read-only operations that are always safe.
_SAFE_PREFIXES: tuple[str, ...] = (
    "get_",
    "list_",
    "read_",
    "search_",
    "fetch_",
    "view_",
    "check_",
    "lookup_",
)


class PolicyChecker:
    """Checks whether a proposed tool call is consistent with the user's task.

    Decision logic:
      1. If the tool name starts with a safe prefix → ALLOW.
      2. If the tool appears in the policy allow-list for a keyword that
         matches the user's task → ALLOW.
      3. If the tool is sensitive AND the user task does not seem to request
         that action → BLOCK.
      4. Otherwise → ALLOW (default-open for non-sensitive tools).
    """

    def __init__(self, config: DefenseConfig) -> None:
        self.config = config
        self._sensitive: set[str] = set(DEFAULT_SENSITIVE_TOOLS)

    # ── public API ─────────────────────────────────────────────────────

    def check(
        self,
        tool_call: ToolCall,
        user_task: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> DefenseVerdict:
        """Return a :class:`DefenseVerdict` for the proposed *tool_call*.

        Args:
            tool_call: The tool call the agent wants to make.
            user_task: The original user request (natural language).
            conversation_history: Optional prior messages (currently unused,
                reserved for future context-aware policies).
        """
        tool_name = tool_call.name.lower()

        # 1. Safe read operations are always allowed.
        if any(tool_name.startswith(p) for p in _SAFE_PREFIXES):
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=1.0,
                explanation=f"Tool '{tool_call.name}' is a read-only operation.",
            )

        # 2. Check the allow-list in DefenseConfig.
        if self._matches_allow_list(tool_name, user_task):
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=0.9,
                explanation=(
                    f"Tool '{tool_call.name}' is allow-listed for the matched "
                    "user-intent keyword."
                ),
            )

        # 3. Sensitive tool without matching user intent → BLOCK.
        if self.is_sensitive(tool_name):
            if not self._intent_matches(tool_name, tool_call.arguments, user_task):
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.85,
                    explanation=(
                        f"Sensitive tool '{tool_call.name}' was called, but the "
                        f"user's task does not appear to request this action. "
                        f"User task: '{user_task[:120]}'"
                    ),
                )

        # 4. Default: allow.
        return DefenseVerdict(
            is_blocked=False,
            layer="policy",
            confidence=0.7,
            explanation=f"Tool '{tool_call.name}' is allowed by default policy.",
        )

    def is_sensitive(self, tool_name: str) -> bool:
        """Return whether *tool_name* is in the sensitive-tools list."""
        return tool_name.lower() in self._sensitive

    # ── internal helpers ───────────────────────────────────────────────

    def _matches_allow_list(self, tool_name: str, user_task: str) -> bool:
        """Check if any keyword in the allow-list matches the user task and
        permits this tool."""
        for keyword, allowed_tools in self.config.policy_allow_list.items():
            if keyword.lower() in user_task.lower():
                if tool_name in (t.lower() for t in allowed_tools):
                    return True
        return False

    @staticmethod
    def _intent_matches(
        tool_name: str,
        arguments: dict[str, Any],
        user_task: str,
    ) -> bool:
        """Simple keyword heuristic to see if the user task plausibly asks
        for the action represented by *tool_name*."""
        task_lower = user_task.lower()

        # Derive keywords from the tool name (split on underscores).
        tool_keywords = tool_name.replace("_", " ").split()
        if any(kw in task_lower for kw in tool_keywords):
            return True

        # Check if argument values appear in the task.
        for value in arguments.values():
            val_str = str(value).lower()
            # Only consider short argument values (likely user-supplied).
            if 3 <= len(val_str) <= 80 and val_str in task_lower:
                return True

        return False
