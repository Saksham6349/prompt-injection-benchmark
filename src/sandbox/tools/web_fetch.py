"""Web fetch tool — retrieve mock web pages from the sandbox database."""

from __future__ import annotations

from src.sandbox.state import SandboxState
from src.types import ToolDefinition, ToolParameter, ToolOutput, AttackChannel


# ── Tool Functions ──────────────────────────────────────────────────────

def fetch_page(state: SandboxState, url: str) -> ToolOutput:
    """Fetch the content of a mock web page by URL."""
    cur = state.conn.cursor()
    cur.execute("SELECT * FROM web_pages WHERE url = ?", (url,))
    row = cur.fetchone()
    if row is None:
        content = f"Error: page not found — {url}"
    else:
        content = (
            f"URL: {row['url']}\n"
            f"Title: {row['title']}\n"
            f"---\n"
            f"{row['content']}"
        )
    return ToolOutput(tool_name="fetch_page", content=content, channel=AttackChannel.WEB_PAGE)


# ── Tool Definitions ───────────────────────────────────────────────────

def get_tool_definitions() -> list[ToolDefinition]:
    """Return OpenAI-compatible tool definitions for the web-fetch tool."""
    return [
        ToolDefinition(
            name="fetch_page",
            description="Fetch the content of a web page by URL.",
            parameters=[
                ToolParameter(name="url", type="string", description="Full URL of the page to fetch"),
            ],
            category="web",
        ),
    ]
