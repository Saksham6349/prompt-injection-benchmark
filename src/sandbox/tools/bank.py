"""Bank API tool — check balances, transfer funds, view transaction history."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.sandbox.state import SandboxState
from src.types import ToolDefinition, ToolParameter, ToolOutput, AttackChannel


# ── Tool Functions ──────────────────────────────────────────────────────

def check_balance(state: SandboxState, account_id: str) -> ToolOutput:
    """Return the current balance of a bank account."""
    cur = state.conn.cursor()
    cur.execute("SELECT * FROM bank_accounts WHERE id = ?", (account_id,))
    row = cur.fetchone()
    if row is None:
        content = f"Account {account_id} not found."
    else:
        content = (
            f"Account: {row['id']}\n"
            f"Owner: {row['owner']}\n"
            f"Balance: {row['balance']:,.2f} {row['currency']}"
        )
    return ToolOutput(tool_name="check_balance", content=content, channel=AttackChannel.API_RESPONSE)


def transfer(
    state: SandboxState,
    from_account: str,
    to_account: str,
    amount: float,
    description: str = "",
) -> ToolOutput:
    """Transfer funds between two accounts."""
    cur = state.conn.cursor()

    # Validate source account
    cur.execute("SELECT balance FROM bank_accounts WHERE id = ?", (from_account,))
    src = cur.fetchone()
    if src is None:
        return ToolOutput(tool_name="transfer", content=f"Source account {from_account} not found.")

    # Validate destination account
    cur.execute("SELECT id FROM bank_accounts WHERE id = ?", (to_account,))
    dst = cur.fetchone()
    if dst is None:
        return ToolOutput(tool_name="transfer", content=f"Destination account {to_account} not found.")

    if amount <= 0:
        return ToolOutput(tool_name="transfer", content="Transfer amount must be positive.")

    if src["balance"] < amount:
        return ToolOutput(
            tool_name="transfer",
            content=f"Insufficient funds. Available: {src['balance']:,.2f}. Requested: {amount:,.2f}.",
        )

    # Execute transfer
    cur.execute("UPDATE bank_accounts SET balance = balance - ? WHERE id = ?", (amount, from_account))
    cur.execute("UPDATE bank_accounts SET balance = balance + ? WHERE id = ?", (amount, to_account))
    cur.execute(
        "INSERT INTO bank_transactions (from_account, to_account, amount, description, timestamp) VALUES (?,?,?,?,?)",
        (from_account, to_account, amount, description, datetime.now(timezone.utc).isoformat()),
    )
    state.conn.commit()

    content = (
        f"Transfer successful.\n"
        f"From: {from_account}\n"
        f"To: {to_account}\n"
        f"Amount: {amount:,.2f}\n"
        f"Description: {description or '(none)'}"
    )
    return ToolOutput(tool_name="transfer", content=content, channel=AttackChannel.API_RESPONSE)


def transaction_history(state: SandboxState, account_id: str) -> ToolOutput:
    """Return recent transactions for an account (sent and received)."""
    cur = state.conn.cursor()
    cur.execute(
        "SELECT * FROM bank_transactions WHERE from_account = ? OR to_account = ? ORDER BY timestamp DESC",
        (account_id, account_id),
    )
    rows = cur.fetchall()
    if not rows:
        content = f"No transactions found for account {account_id}."
    else:
        lines = [f"Transaction history for {account_id} ({len(rows)} entries):\n"]
        for r in rows:
            direction = "SENT" if r["from_account"] == account_id else "RECEIVED"
            other = r["to_account"] if direction == "SENT" else r["from_account"]
            lines.append(
                f"  [{direction}] {r['amount']:,.2f} {'to' if direction == 'SENT' else 'from'} {other}"
                f" — {r['description']} ({r['timestamp']})"
            )
        content = "\n".join(lines)
    return ToolOutput(tool_name="transaction_history", content=content, channel=AttackChannel.API_RESPONSE)


# ── Tool Definitions ───────────────────────────────────────────────────

def get_tool_definitions() -> list[ToolDefinition]:
    """Return OpenAI-compatible tool definitions for bank tools."""
    return [
        ToolDefinition(
            name="check_balance",
            description="Check the current balance of a bank account.",
            parameters=[
                ToolParameter(name="account_id", type="string", description="Bank account ID (e.g. 'ACC-1001')"),
            ],
            category="bank",
        ),
        ToolDefinition(
            name="transfer",
            description="Transfer money from one bank account to another.",
            parameters=[
                ToolParameter(name="from_account", type="string", description="Source account ID"),
                ToolParameter(name="to_account", type="string", description="Destination account ID"),
                ToolParameter(name="amount", type="number", description="Amount to transfer"),
                ToolParameter(name="description", type="string", description="Transfer description/memo", required=False),
            ],
            category="bank",
        ),
        ToolDefinition(
            name="transaction_history",
            description="Get recent transaction history for a bank account.",
            parameters=[
                ToolParameter(name="account_id", type="string", description="Bank account ID"),
            ],
            category="bank",
        ),
    ]
