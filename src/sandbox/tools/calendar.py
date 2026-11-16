"""Calendar tool — list, create, update, and delete mock calendar events."""

from __future__ import annotations

from typing import Any

from src.sandbox.state import SandboxState
from src.types import ToolDefinition, ToolParameter, ToolOutput


# ── Tool Functions ──────────────────────────────────────────────────────

def list_events(state: SandboxState, user: str | None = None, date_range: str | None = None) -> ToolOutput:
    """List calendar events, optionally filtered by date range.

    *date_range* should be ``"YYYY-MM-DD to YYYY-MM-DD"`` or a single date.
    *user* is accepted for API symmetry but the mock calendar is shared.
    """
    cur = state.conn.cursor()
    if date_range:
        parts = [d.strip() for d in date_range.replace("to", "|").replace("-", "-").split("|")]
        if len(parts) == 2:
            cur.execute(
                "SELECT * FROM calendar_events WHERE date BETWEEN ? AND ? ORDER BY date, time",
                (parts[0], parts[1]),
            )
        else:
            cur.execute(
                "SELECT * FROM calendar_events WHERE date = ? ORDER BY time",
                (parts[0],),
            )
    else:
        cur.execute("SELECT * FROM calendar_events ORDER BY date, time")
    rows = cur.fetchall()
    if not rows:
        content = "No calendar events found."
    else:
        lines = [f"Calendar events ({len(rows)}):\n"]
        for r in rows:
            lines.append(
                f"  #{r['id']} {r['date']} {r['time']} ({r['duration']}min) — {r['title']}"
                f"  @ {r['location']}"
            )
        content = "\n".join(lines)
    return ToolOutput(tool_name="list_events", content=content)


def get_event(state: SandboxState, event_id: int) -> ToolOutput:
    """Return full details for a single calendar event."""
    cur = state.conn.cursor()
    cur.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,))
    row = cur.fetchone()
    if row is None:
        content = f"Event #{event_id} not found."
    else:
        content = (
            f"Event #{row['id']}\n"
            f"Title: {row['title']}\n"
            f"Date: {row['date']}\n"
            f"Time: {row['time']}\n"
            f"Duration: {row['duration']} minutes\n"
            f"Location: {row['location']}\n"
            f"Description: {row['description']}"
        )
    return ToolOutput(tool_name="get_event", content=content)


def create_event(
    state: SandboxState,
    title: str,
    date: str,
    time: str,
    duration: int = 60,
    location: str = "",
    description: str = "",
) -> ToolOutput:
    """Create a new calendar event and return its ID."""
    cur = state.conn.cursor()
    cur.execute(
        "INSERT INTO calendar_events (title, date, time, duration, location, description) VALUES (?,?,?,?,?,?)",
        (title, date, time, duration, location, description),
    )
    state.conn.commit()
    event_id = cur.lastrowid
    content = f"Event created: #{event_id} '{title}' on {date} at {time}."
    return ToolOutput(tool_name="create_event", content=content)


def update_event(state: SandboxState, event_id: int, **kwargs: Any) -> ToolOutput:
    """Update one or more fields of an existing calendar event."""
    allowed = {"title", "date", "time", "duration", "location", "description"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return ToolOutput(tool_name="update_event", content="No valid fields to update.")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [event_id]
    cur = state.conn.cursor()
    cur.execute(f"UPDATE calendar_events SET {set_clause} WHERE id = ?", values)
    state.conn.commit()
    if cur.rowcount == 0:
        content = f"Event #{event_id} not found."
    else:
        content = f"Event #{event_id} updated: {', '.join(f'{k}={v!r}' for k, v in updates.items())}."
    return ToolOutput(tool_name="update_event", content=content)


def delete_event(state: SandboxState, event_id: int) -> ToolOutput:
    """Delete a calendar event by ID."""
    cur = state.conn.cursor()
    cur.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))
    state.conn.commit()
    if cur.rowcount == 0:
        content = f"Event #{event_id} not found."
    else:
        content = f"Event #{event_id} deleted."
    return ToolOutput(tool_name="delete_event", content=content)


# ── Tool Definitions ───────────────────────────────────────────────────

def get_tool_definitions() -> list[ToolDefinition]:
    """Return OpenAI-compatible tool definitions for calendar tools."""
    return [
        ToolDefinition(
            name="list_events",
            description="List calendar events, optionally filtered by date range.",
            parameters=[
                ToolParameter(name="user", type="string", description="Username (optional filter)", required=False),
                ToolParameter(name="date_range", type="string", description="Date range, e.g. '2026-11-15 to 2026-11-20' or a single date", required=False),
            ],
            category="calendar",
        ),
        ToolDefinition(
            name="get_event",
            description="Get full details of a calendar event by ID.",
            parameters=[
                ToolParameter(name="event_id", type="integer", description="Numeric event ID"),
            ],
            category="calendar",
        ),
        ToolDefinition(
            name="create_event",
            description="Create a new calendar event.",
            parameters=[
                ToolParameter(name="title", type="string", description="Event title"),
                ToolParameter(name="date", type="string", description="Date in YYYY-MM-DD format"),
                ToolParameter(name="time", type="string", description="Time in HH:MM format"),
                ToolParameter(name="duration", type="integer", description="Duration in minutes", required=False),
                ToolParameter(name="location", type="string", description="Event location", required=False),
                ToolParameter(name="description", type="string", description="Event description", required=False),
            ],
            category="calendar",
        ),
        ToolDefinition(
            name="update_event",
            description="Update fields of an existing calendar event.",
            parameters=[
                ToolParameter(name="event_id", type="integer", description="Numeric event ID"),
                ToolParameter(name="title", type="string", description="New title", required=False),
                ToolParameter(name="date", type="string", description="New date (YYYY-MM-DD)", required=False),
                ToolParameter(name="time", type="string", description="New time (HH:MM)", required=False),
                ToolParameter(name="duration", type="integer", description="New duration in minutes", required=False),
                ToolParameter(name="location", type="string", description="New location", required=False),
                ToolParameter(name="description", type="string", description="New description", required=False),
            ],
            category="calendar",
        ),
        ToolDefinition(
            name="delete_event",
            description="Delete a calendar event by ID.",
            parameters=[
                ToolParameter(name="event_id", type="integer", description="Numeric event ID"),
            ],
            category="calendar",
        ),
    ]
