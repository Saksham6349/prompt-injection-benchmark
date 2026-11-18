"""Layer 3: Hardened Tool-Call Policy Checker (Phase 2).

Enforces pre-execution tool authorization and capability boundaries:
1. Tool Capability Manifest: Default-deny for unknown mutating tools; argument/host checks for fetch_page.
2. Structured Intent Matching: Identifies authorized action categories; handles negation and questions.
3. Argument Binding: Validates that tool call arguments (e.g. email recipient, transfer account)
   match user intent or trusted state, defeating argument hijacking.
"""

from __future__ import annotations

import re
import sys
import os
from typing import Any, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.types import DefenseConfig, DefenseVerdict, ToolCall
from src.defense.manifest import (
    ToolCategory,
    get_tool_category,
    is_tool_sensitive,
    validate_network_tool_args,
    TOOL_MANIFEST,
)

# Read-only operations that are safe
_SAFE_PREFIXES: tuple[str, ...] = (
    "get_",
    "list_",
    "read_",
    "search_",
    "view_",
    "check_",
    "lookup_",
)

# Canonical intent mapping from action semantic verbs to allowed tool families
INTENT_TOOL_MAPPING: dict[str, set[str]] = {
    "email": {"send_email", "send_message", "read_email", "list_emails", "get_email"},
    "send_email": {"send_email", "send_message"},
    "transfer": {"transfer", "bank_transfer", "transfer_money"},
    "delete_file": {"delete_file"},
    "delete_event": {"delete_event"},
    "update_event": {"update_event"},
    "create_event": {"create_calendar_event"},
}

# Negation patterns indicating explicit prohibition
NEGATION_PATTERNS = [
    r"\b(do\s+not|don't|dont|never|should\s+not|shouldn't|must\s+not|mustn't)\b",
    r"\b(cannot|can't|prohibit|prohibited|under\s+no\s+circumstances|refrain\s+from|stop)\b",
    r"\b(no\s+(?:email|transfer|deletion|message|payment))\b",
    r"\b(not\s+(?:send|transfer|delete|execute|wire|pay))\b",
]

# Question / query patterns indicating an inquiry rather than an execution command
QUESTION_PATTERNS = [
    r"^\s*(did|have|has|was|were|is|are)\s+(i|you|we|any|the)\b",
    r"^\s*(can|could|would)\s+you\s+(check|verify|see|confirm)\s+(if|whether)\b",
    r"\b(did\s+(?:i|you|we)\s+(?:already\s+)?(?:send|transfer|delete|pay|wire))\b",
    r"\b(was\s+(?:an?\s+)?(?:email|transfer|message|file)\s+(?:sent|deleted|made))\b",
]


class PolicyChecker:
    """Checks whether a proposed tool call is consistent with the user's task.

    Decision logic (Hardened Phase 2):
      1. If tool is fetch_page (network read), validate URL, scheme, and SSRF constraints.
      2. If tool is read-only safe operation → ALLOW.
      3. If tool is unknown and mutating → BLOCK (Default-Deny for unknown mutating tools).
      4. If tool is sensitive/mutating:
         a. Check for explicit negation in user task → BLOCK.
         b. Check for question inquiry rather than command → BLOCK.
         c. Verify structured action authorization → BLOCK if intent missing.
         d. Verify argument binding against user task or trusted state → BLOCK if arguments hijacked.
      5. Otherwise → ALLOW.
    """

    def __init__(self, config: DefenseConfig) -> None:
        self.config = config
        self.mode = getattr(config, "policy_mode", "hardened")

    # ── public API ─────────────────────────────────────────────────────

    def check(
        self,
        tool_call: ToolCall,
        user_task: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> DefenseVerdict:
        """Return a :class:`DefenseVerdict` for the proposed *tool_call*."""
        tool_name = tool_call.name.lower().strip()

        # Legacy Mode Support (Phase 1 baseline preservation)
        if self.mode == "legacy":
            return self._check_legacy(tool_name, tool_call.arguments, user_task)

        # ── 1. Network Read Validation (fetch_page SSRF / Host constraint) ──
        if tool_name == "fetch_page" or "fetch" in tool_name:
            valid, reason = validate_network_tool_args(tool_name, tool_call.arguments)
            if not valid:
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.98,
                    explanation=f"Network argument validation failed for '{tool_call.name}': {reason}",
                )
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=1.0,
                explanation=f"Tool '{tool_call.name}' validated under network read security constraints.",
            )

        # ── 2. Safe Read-Only Operations ──
        category = get_tool_category(tool_name)
        if category == ToolCategory.READ_ONLY or any(tool_name.startswith(p) for p in _SAFE_PREFIXES):
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=1.0,
                explanation=f"Tool '{tool_call.name}' is a registered read-only operation.",
            )

        # ── 3. Default-Deny on Unknown Mutating / Unregistered Tools ──
        if tool_name not in TOOL_MANIFEST:
            if category in (ToolCategory.MUTATING, ToolCategory.UNKNOWN):
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.98,
                    explanation=(
                        f"Default-Deny: Tool '{tool_call.name}' is an unregistered mutating tool. "
                        "Mutating tools not declared in the tool capability manifest are strictly denied."
                    ),
                )
            if category == ToolCategory.UNKNOWN:
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.95,
                    explanation=(
                        f"Default-Deny: Tool '{tool_call.name}' is not registered in the tool capability manifest."
                    ),
                )

        # ── 4. Explicit Allow-List in DefenseConfig ──
        if self._matches_allow_list(tool_name, user_task):
            # Still verify arguments even if allow-listed
            bound, arg_reason = self._verify_argument_binding(tool_name, tool_call.arguments, user_task)
            if not bound:
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.92,
                    explanation=f"Argument Binding Violation: {arg_reason}",
                )
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=0.95,
                explanation=f"Tool '{tool_call.name}' is explicitly allow-listed with verified argument binding.",
            )

        # ── 5. Sensitive / Mutating Tool Authorization & Argument Binding ──
        if self.is_sensitive(tool_name):
            # (a) Check Negation Handling
            if self._is_action_negated(tool_name, user_task):
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.95,
                    explanation=(
                        f"Negation Protection: Action '{tool_call.name}' was explicitly negated or prohibited "
                        f"by the user task ('{user_task[:120]}')."
                    ),
                )

            # (b) Check Question / Query Handling
            if self._is_query_question(tool_name, user_task):
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.90,
                    explanation=(
                        f"Question Intent Protection: User task is an inquiry/question, which does not "
                        f"authorize executing mutating tool '{tool_call.name}'."
                    ),
                )

            # (c) Check Structured Intent Match
            if not self._intent_matches(tool_name, tool_call.arguments, user_task):
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.90,
                    explanation=(
                        f"Intent Authorization Failure: Sensitive tool '{tool_call.name}' was invoked, but "
                        f"the user's task does not request this action. User task: '{user_task[:120]}'"
                    ),
                )

            # (d) Check Argument Binding against User Task
            is_bound, bind_reason = self._verify_argument_binding(tool_name, tool_call.arguments, user_task)
            if not is_bound:
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.92,
                    explanation=f"Argument Hijacking Protection: {bind_reason}",
                )

            # All checks passed for sensitive tool
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=0.90,
                explanation=f"Sensitive tool '{tool_call.name}' authorized by user task with verified argument binding.",
            )

        # Default open for safe tools
        return DefenseVerdict(
            is_blocked=False,
            layer="policy",
            confidence=0.8,
            explanation=f"Tool '{tool_call.name}' is permitted by default policy.",
        )

    def is_sensitive(self, tool_name: str) -> bool:
        """Return whether *tool_name* is sensitive or mutating."""
        return is_tool_sensitive(tool_name)

    # ── internal verification logic ────────────────────────────────────

    def _is_action_negated(self, tool_name: str, user_task: str) -> bool:
        """Return True if the action is explicitly negated in user_task."""
        task_lower = user_task.lower()
        action_stems = self._get_action_stems(tool_name)

        tight_neg_cues = [
            r"do\s+not",
            r"don't",
            r"dont",
            r"never",
            r"must\s+not",
            r"mustn't",
            r"should\s+not",
            r"shouldn't",
            r"cannot",
            r"can't",
            r"refrain\s+from",
            r"stop",
            r"prohibit(?:ed)?",
            r"under\s+no\s+circumstances",
        ]

        for stem in action_stems:
            for cue in tight_neg_cues:
                # Proximity: up to 3 intervening words
                pattern = rf"\b{cue}\b(?:\s+\w+){{0,3}}\s+\b{stem}\b"
                if re.search(pattern, task_lower):
                    return True
            # Pattern like "no email to", "no transfer"
            pattern_no = rf"\bno\s+(?:any\s+)?\b{stem}\b"
            if re.search(pattern_no, task_lower):
                return True
        return False

    def _is_query_question(self, tool_name: str, user_task: str) -> bool:
        """Return True if user_task is a question/inquiry rather than a command."""
        clean_task = user_task.strip()
        task_lower = clean_task.lower()

        # Tasks ending with ? or starting with question auxiliaries
        is_question_syntax = clean_task.endswith("?") or any(re.search(pat, task_lower) for pat in QUESTION_PATTERNS)
        if not is_question_syntax:
            return False

        # Exclude polite imperative requests that end with question mark
        # e.g., "Could you please send an email to alice@example.com?"
        imperative_polite_cues = [
            r"^(could|can|would)\s+you\s+(?:please\s+)?(send|transfer|delete|update|create|write)",
            r"^(please\s+)?(send|transfer|delete|update|create|write)",
        ]
        if any(re.search(pat, task_lower) for pat in imperative_polite_cues):
            # This is a polite request/command, NOT an inquiry about past actions
            return False

        return True

    def _intent_matches(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_task: str,
    ) -> bool:
        """Structured intent match verifying that the user explicitly authorized this action."""
        task_lower = user_task.lower()
        stems = self._get_action_stems(tool_name)
        
        # Check if any legitimate action verb appears in user task
        return any(re.search(rf"\b{stem}\b", task_lower) for stem in stems)

    def _verify_argument_binding(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_task: str,
    ) -> tuple[bool, str]:
        """Verify that crucial tool arguments match entities specified in user_task."""
        task_lower = user_task.lower()

        # 1. Email Recipient Binding
        if tool_name in ("send_email", "send_message"):
            to_arg = str(arguments.get("to") or arguments.get("recipient") or "").strip().lower()
            if to_arg:
                # Find all email addresses in user task
                user_emails = [e.lower() for e in re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", user_task)]
                if user_emails:
                    if to_arg not in user_emails:
                        return False, f"Target recipient '{to_arg}' does not match any recipient in user task: {user_emails}"
                else:
                    # Check if username or name alias appears in task
                    name_prefix = to_arg.split("@")[0]
                    if name_prefix not in task_lower and to_arg not in task_lower:
                        return False, f"Target recipient '{to_arg}' was not specified or authorized in user task"

        # 2. Bank Transfer Account & Amount Binding
        if tool_name in ("transfer", "bank_transfer", "transfer_money"):
            to_acc = str(arguments.get("to_account") or arguments.get("account") or "").strip()
            if to_acc:
                # Look for account identifiers in task (ACC-..., numbers)
                if to_acc.lower() not in task_lower:
                    # Also look for normalized representation without hyphens
                    clean_acc = to_acc.replace("-", "").lower()
                    if clean_acc not in task_lower.replace("-", ""):
                        return False, f"Destination account '{to_acc}' does not match authorized account in user task"

            amt_arg = arguments.get("amount")
            if amt_arg is not None:
                try:
                    num_amt = float(amt_arg)
                    # Extract any dollar/rupee/number amounts from user task
                    amounts_in_task = [float(x) for x in re.findall(r"\$?\b(\d+(?:\.\d+)?)\b", user_task)]
                    if amounts_in_task:
                        max_auth = max(amounts_in_task)
                        if num_amt > max_auth:
                            return False, f"Transfer amount ${num_amt:.2f} exceeds user-authorized limit of ${max_auth:.2f}"
                except (ValueError, TypeError):
                    pass

        # 3. File Deletion Target Binding
        if tool_name == "delete_file":
            path_arg = str(arguments.get("path") or arguments.get("filename") or arguments.get("file") or "").strip().lower()
            if path_arg:
                # Extract basename
                basename = path_arg.split("/")[-1].split("\\")[-1]
                if basename not in task_lower and path_arg not in task_lower:
                    return False, f"Target file '{path_arg}' was not specified for deletion in user task"

        # 4. Event Deletion Target Binding
        if tool_name in ("delete_event", "update_event"):
            evt_arg = str(arguments.get("event_id") or arguments.get("id") or "").strip().lower()
            if evt_arg and len(evt_arg) >= 3:
                if evt_arg not in task_lower:
                    # If user task specified a specific event ID, it must match
                    task_event_ids = re.findall(r"\b(evt-\d+|event_\d+|\d+)\b", task_lower)
                    if task_event_ids and evt_arg not in task_event_ids:
                        return False, f"Target event ID '{evt_arg}' does not match event authorized in user task"

        return True, "Arguments correctly bound to user task intent"

    @staticmethod
    def _get_action_stems(tool_name: str) -> list[str]:
        """Return search stems and inflections for tool actions."""
        if "email" in tool_name or "message" in tool_name:
            return [
                "email", "emails", "emailed", "emailing",
                "send", "sent", "sending",
                "mail", "mailed", "mailing",
                "shoot", "message", "reply"
            ]
        if "transfer" in tool_name or "bank" in tool_name or "pay" in tool_name:
            return [
                "transfer", "transferred", "transferring", "transfers",
                "wire", "wired", "wiring",
                "pay", "paid", "paying",
                "send", "sent", "sending",
                "settle", "settled",
                "remit", "remitted", "remitting"
            ]
        if "delete_file" in tool_name:
            return [
                "delete", "deleted", "deleting", "deletion",
                "remove", "removed", "removing", "removal",
                "erase", "erased", "erasing", "erasure",
                "drop", "dropped", "dropping",
                "purge", "purged", "purging"
            ]
        if "delete_event" in tool_name:
            return [
                "delete", "deleted", "deleting", "deletion",
                "cancel", "cancelled", "canceled", "cancelling", "cancellation",
                "remove", "removed", "removing",
                "drop", "dropped"
            ]
        if "update_event" in tool_name:
            return [
                "update", "updated", "updating",
                "reschedule", "rescheduled", "rescheduling",
                "modify", "modified", "modifying",
                "change", "changed", "move", "moved"
            ]
        if "calendar" in tool_name or "event" in tool_name:
            return [
                "schedule", "scheduled", "create", "created", "book", "booked", "add", "added"
            ]
        return tool_name.replace("_", " ").split()

    def _matches_allow_list(self, tool_name: str, user_task: str) -> bool:
        """Check if any keyword in the allow-list permits this tool."""
        for keyword, allowed_tools in self.config.policy_allow_list.items():
            if keyword.lower() in user_task.lower():
                if tool_name in (t.lower() for t in allowed_tools):
                    return True
        return False

    # ── Legacy Baseline Checker ────────────────────────────────────────

    def _check_legacy(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_task: str,
    ) -> DefenseVerdict:
        """Phase 1 legacy implementation preserved for regression comparison."""
        if any(tool_name.startswith(p) for p in _SAFE_PREFIXES):
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=1.0,
                explanation=f"Tool '{tool_name}' is a read-only operation.",
            )

        if self._matches_allow_list(tool_name, user_task):
            return DefenseVerdict(
                is_blocked=False,
                layer="policy",
                confidence=0.9,
                explanation=f"Tool '{tool_name}' is allow-listed for matched user-intent keyword.",
            )

        if self.is_sensitive(tool_name):
            task_lower = user_task.lower()
            tool_keywords = tool_name.replace("_", " ").split()
            matched = any(kw in task_lower for kw in tool_keywords)
            if not matched:
                for value in arguments.values():
                    val_str = str(value).lower()
                    if 3 <= len(val_str) <= 80 and val_str in task_lower:
                        matched = True
                        break
            if not matched:
                return DefenseVerdict(
                    is_blocked=True,
                    layer="policy",
                    confidence=0.85,
                    explanation=f"Sensitive tool '{tool_name}' was called, but user task does not request action.",
                )

        return DefenseVerdict(
            is_blocked=False,
            layer="policy",
            confidence=0.7,
            explanation=f"Tool '{tool_name}' is allowed by default policy.",
        )
