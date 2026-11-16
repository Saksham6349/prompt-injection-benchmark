"""File store tool — list, read, create, delete, and search mock files."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.sandbox.state import SandboxState
from src.types import ToolDefinition, ToolParameter, ToolOutput, AttackChannel


# ── Tool Functions ──────────────────────────────────────────────────────

def list_files(state: SandboxState, owner: str) -> ToolOutput:
    """List all files belonging to *owner*."""
    cur = state.conn.cursor()
    cur.execute(
        "SELECT id, name, owner, created_at FROM files WHERE owner = ? ORDER BY name",
        (owner,),
    )
    rows = cur.fetchall()
    if not rows:
        content = f"No files found for owner '{owner}'."
    else:
        lines = [f"Files owned by {owner} ({len(rows)}):\n"]
        for r in rows:
            lines.append(f"  #{r['id']} {r['name']}  (created {r['created_at']})")
        content = "\n".join(lines)
    return ToolOutput(tool_name="list_files", content=content, channel=AttackChannel.RETRIEVED_DOCUMENT)


def read_file_tool(state: SandboxState, file_id: int) -> ToolOutput:
    """Return the full content of a file by ID."""
    cur = state.conn.cursor()
    cur.execute("SELECT * FROM files WHERE id = ?", (file_id,))
    row = cur.fetchone()
    if row is None:
        content = f"File #{file_id} not found."
    else:
        content = (
            f"File #{row['id']}: {row['name']}\n"
            f"Owner: {row['owner']}\n"
            f"Created: {row['created_at']}\n"
            f"---\n"
            f"{row['content']}"
        )
    return ToolOutput(tool_name="read_file", content=content, channel=AttackChannel.RETRIEVED_DOCUMENT)


def create_file(state: SandboxState, name: str, content: str, owner: str) -> ToolOutput:
    """Create a new file in the store."""
    cur = state.conn.cursor()
    cur.execute(
        "INSERT INTO files (name, content, owner, created_at) VALUES (?,?,?,?)",
        (name, content, owner, datetime.utcnow().isoformat()),
    )
    state.conn.commit()
    file_id = cur.lastrowid
    return ToolOutput(
        tool_name="create_file",
        content=f"File created: #{file_id} '{name}' (owner: {owner}).",
        channel=AttackChannel.RETRIEVED_DOCUMENT,
    )


def delete_file(state: SandboxState, file_id: int) -> ToolOutput:
    """Delete a file by ID."""
    cur = state.conn.cursor()
    cur.execute("DELETE FROM files WHERE id = ?", (file_id,))
    state.conn.commit()
    if cur.rowcount == 0:
        content = f"File #{file_id} not found."
    else:
        content = f"File #{file_id} deleted."
    return ToolOutput(tool_name="delete_file", content=content)


def search_files(state: SandboxState, query: str) -> ToolOutput:
    """Search files by keyword in name or content."""
    cur = state.conn.cursor()
    like = f"%{query}%"
    cur.execute(
        "SELECT id, name, owner, created_at FROM files WHERE name LIKE ? OR content LIKE ? ORDER BY name",
        (like, like),
    )
    rows = cur.fetchall()
    if not rows:
        content = f"No files matching '{query}'."
    else:
        lines = [f"Search results for '{query}' ({len(rows)} matches):\n"]
        for r in rows:
            lines.append(f"  #{r['id']} {r['name']} (owner: {r['owner']})")
        content = "\n".join(lines)
    return ToolOutput(
        tool_name="search_files",
        content=content,
        channel=AttackChannel.RETRIEVED_DOCUMENT,
    )


# ── Tool Definitions ───────────────────────────────────────────────────

def get_tool_definitions() -> list[ToolDefinition]:
    """Return OpenAI-compatible tool definitions for file-store tools."""
    return [
        ToolDefinition(
            name="list_files",
            description="List all files belonging to a user.",
            parameters=[
                ToolParameter(name="owner", type="string", description="File owner username"),
            ],
            category="filestore",
        ),
        ToolDefinition(
            name="read_file",
            description="Read the full content of a file by its ID.",
            parameters=[
                ToolParameter(name="file_id", type="integer", description="Numeric file ID"),
            ],
            category="filestore",
        ),
        ToolDefinition(
            name="create_file",
            description="Create a new file with given name, content, and owner.",
            parameters=[
                ToolParameter(name="name", type="string", description="File name"),
                ToolParameter(name="content", type="string", description="File content"),
                ToolParameter(name="owner", type="string", description="File owner username"),
            ],
            category="filestore",
        ),
        ToolDefinition(
            name="delete_file",
            description="Delete a file by its ID.",
            parameters=[
                ToolParameter(name="file_id", type="integer", description="Numeric file ID"),
            ],
            category="filestore",
        ),
        ToolDefinition(
            name="search_files",
            description="Search files by keyword in name or content.",
            parameters=[
                ToolParameter(name="query", type="string", description="Search keyword or phrase"),
            ],
            category="filestore",
        ),
    ]
