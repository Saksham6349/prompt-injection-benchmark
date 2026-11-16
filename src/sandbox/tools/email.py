"""Email tool — read, send, and search mock emails."""

from __future__ import annotations

from typing import Any

from src.sandbox.state import SandboxState
from src.types import ToolDefinition, ToolParameter, ToolOutput, AttackChannel


# ── Tool Functions ──────────────────────────────────────────────────────

def read_inbox(state: SandboxState, user: str) -> ToolOutput:
    """Return all emails for *user* (as recipient), newest first."""
    cur = state.conn.cursor()
    cur.execute(
        "SELECT id, sender, subject, timestamp, read FROM emails WHERE recipient = ? ORDER BY timestamp DESC",
        (f"{user}@company.com",),
    )
    rows = cur.fetchall()
    if not rows:
        content = f"No emails found for {user}."
    else:
        lines = [f"Inbox for {user} ({len(rows)} messages):\n"]
        for r in rows:
            status = "READ" if r["read"] else "UNREAD"
            lines.append(
                f"  [{status}] #{r['id']} From: {r['sender']} | Subject: {r['subject']} | {r['timestamp']}"
            )
        content = "\n".join(lines)
    return ToolOutput(tool_name="read_inbox", content=content, channel=AttackChannel.EMAIL)


def read_email(state: SandboxState, email_id: int) -> ToolOutput:
    """Return the full body of a single email and mark it as read."""
    cur = state.conn.cursor()
    cur.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
    row = cur.fetchone()
    if row is None:
        content = f"Email #{email_id} not found."
    else:
        cur.execute("UPDATE emails SET read = 1 WHERE id = ?", (email_id,))
        state.conn.commit()
        content = (
            f"Email #{row['id']}\n"
            f"From: {row['sender']}\n"
            f"To: {row['recipient']}\n"
            f"Subject: {row['subject']}\n"
            f"Date: {row['timestamp']}\n"
            f"---\n"
            f"{row['body']}"
        )
    return ToolOutput(tool_name="read_email", content=content, channel=AttackChannel.EMAIL)


def send_email(state: SandboxState, to: str, subject: str, body: str) -> ToolOutput:
    """Compose and store a new email from the agent's default account."""
    from datetime import datetime

    cur = state.conn.cursor()
    cur.execute(
        "INSERT INTO emails (sender, recipient, subject, body, read, timestamp) VALUES (?,?,?,?,?,?)",
        ("agent@company.com", to, subject, body, 0, datetime.utcnow().isoformat()),
    )
    state.conn.commit()
    content = f"Email sent to {to} with subject '{subject}'."
    return ToolOutput(tool_name="send_email", content=content, channel=AttackChannel.EMAIL)


def search_emails(state: SandboxState, query: str) -> ToolOutput:
    """Search emails by keyword across subject and body fields."""
    cur = state.conn.cursor()
    like = f"%{query}%"
    cur.execute(
        "SELECT id, sender, recipient, subject, timestamp FROM emails WHERE subject LIKE ? OR body LIKE ? ORDER BY timestamp DESC",
        (like, like),
    )
    rows = cur.fetchall()
    if not rows:
        content = f"No emails matching '{query}'."
    else:
        lines = [f"Search results for '{query}' ({len(rows)} matches):\n"]
        for r in rows:
            lines.append(
                f"  #{r['id']} From: {r['sender']} → {r['recipient']} | {r['subject']} | {r['timestamp']}"
            )
        content = "\n".join(lines)
    return ToolOutput(tool_name="search_emails", content=content, channel=AttackChannel.EMAIL)


# ── Tool Definitions ───────────────────────────────────────────────────

def get_tool_definitions() -> list[ToolDefinition]:
    """Return OpenAI-compatible tool definitions for email tools."""
    return [
        ToolDefinition(
            name="read_inbox",
            description="List all emails in a user's inbox, showing sender, subject, and read status.",
            parameters=[
                ToolParameter(name="user", type="string", description="Username (e.g. 'alice')"),
            ],
            category="email",
        ),
        ToolDefinition(
            name="read_email",
            description="Retrieve the full content of an email by its ID.",
            parameters=[
                ToolParameter(name="email_id", type="integer", description="Numeric email ID"),
            ],
            category="email",
        ),
        ToolDefinition(
            name="send_email",
            description="Send a new email to a recipient.",
            parameters=[
                ToolParameter(name="to", type="string", description="Recipient email address"),
                ToolParameter(name="subject", type="string", description="Email subject line"),
                ToolParameter(name="body", type="string", description="Email body text"),
            ],
            category="email",
        ),
        ToolDefinition(
            name="search_emails",
            description="Search emails by keyword in subject or body.",
            parameters=[
                ToolParameter(name="query", type="string", description="Search keyword or phrase"),
            ],
            category="email",
        ),
    ]
