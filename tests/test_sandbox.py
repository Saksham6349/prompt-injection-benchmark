"""Unit tests for the sandbox tools and state management."""

from __future__ import annotations

from src.sandbox.state import SandboxState
from src.sandbox.executor import ToolExecutor
from src.types import ToolCall


def test_sandbox_state_initialization():
    state = SandboxState()
    snap = state.get_state_snapshot()
    assert "emails" in snap
    assert "calendar_events" in snap
    assert "files" in snap
    assert "bank_accounts" in snap
    assert "web_pages" in snap
    assert len(snap["bank_accounts"]) > 0


def test_email_tool_read_and_send():
    state = SandboxState()
    executor = ToolExecutor(state)

    # Read inbox
    out = executor.execute("read_inbox", {"user": "alice"})
    assert "alice" in out.content.lower() or "subject" in out.content.lower()

    # Send email
    send_out = executor.execute(
        "send_email",
        {"to": "bob@example.com", "subject": "Test", "body": "Hello Bob"},
    )
    assert "sent" in send_out.content.lower()

    # Check state updated
    snap = state.get_state_snapshot()
    sent_emails = [e for e in snap["emails"] if e["recipient"] == "bob@example.com"]
    assert len(sent_emails) == 1


def test_bank_tool_transfer():
    state = SandboxState()
    executor = ToolExecutor(state)

    # Check initial balance
    bal_out = executor.execute("check_balance", {"account_id": "ACC-1001"})
    assert "12,500.00" in bal_out.content

    # Transfer
    tx_out = executor.execute(
        "transfer",
        {"from_account": "ACC-1001", "to_account": "ACC-1002", "amount": 200, "description": "Gift"},
    )
    assert "transferred" in tx_out.content.lower() or "success" in tx_out.content.lower()

    # Check balance after
    bal_out2 = executor.execute("check_balance", {"account_id": "ACC-1001"})
    assert "12,300.00" in bal_out2.content


def test_sandbox_reset():
    state = SandboxState()
    executor = ToolExecutor(state)

    # Mutate state
    executor.execute("create_file", {"name": "temp.txt", "content": "123", "owner": "alice"})
    snap_before = state.get_state_snapshot()
    assert any(f["name"] == "temp.txt" for f in snap_before["files"])

    # Reset
    executor.reset()
    snap_after = state.get_state_snapshot()
    assert not any(f["name"] == "temp.txt" for f in snap_after["files"])
