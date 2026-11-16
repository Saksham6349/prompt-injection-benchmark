"""SQLite-based response cache for LLM API calls.

Makes benchmark runs reproducible and reduces API costs by caching
provider responses keyed on (provider, model, messages_hash, tools_hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.types import ToolCall

from .providers.base import ProviderResponse

logger = logging.getLogger(__name__)


class ResponseCache:
    """Persistent cache for LLM provider responses."""

    def __init__(self, db_path: str = "data/response_cache.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_table()

    # ── public ──────────────────────────────────────────────────────

    def get(self, provider: str, model: str, messages: list[dict], tools: list[dict]) -> ProviderResponse | None:
        """Look up a cached response.

        Returns:
            A ProviderResponse if a cache hit is found, else None.
        """
        key = self._make_key(provider, model, messages, tools)
        cur = self._conn.execute(
            "SELECT response_json FROM response_cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        logger.debug("Cache HIT for key=%s", key[:16])
        return self._deserialize(row[0])

    def put(self, provider: str, model: str, messages: list[dict], tools: list[dict], response: ProviderResponse) -> None:
        """Store a response in the cache (upsert)."""
        key = self._make_key(provider, model, messages, tools)
        blob = self._serialize(response)
        self._conn.execute(
            "INSERT OR REPLACE INTO response_cache (cache_key, provider, model, response_json) VALUES (?, ?, ?, ?)",
            (key, provider, model, blob),
        )
        self._conn.commit()
        logger.debug("Cache PUT for key=%s", key[:16])

    def clear(self) -> int:
        """Delete all cached entries. Returns number of rows removed."""
        cur = self._conn.execute("DELETE FROM response_cache")
        self._conn.commit()
        count = cur.rowcount
        logger.info("Cache cleared: %d entries removed", count)
        return count

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── private ─────────────────────────────────────────────────────

    def _init_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key   TEXT PRIMARY KEY,
                provider    TEXT NOT NULL,
                model       TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    @staticmethod
    def _make_key(provider: str, model: str, messages: list[dict], tools: list[dict]) -> str:
        """Deterministic hash key from the request parameters."""
        # Canonicalise by sorting JSON keys
        blob = json.dumps({
            "provider": provider,
            "model": model,
            "messages": messages,
            "tools": tools,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _serialize(resp: ProviderResponse) -> str:
        """Serialise a ProviderResponse to JSON (drops raw_response)."""
        tool_calls_data = []
        for tc in resp.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })
        return json.dumps({
            "content": resp.content,
            "tool_calls": tool_calls_data,
            "usage": resp.usage,
        })

    @staticmethod
    def _deserialize(blob: str) -> ProviderResponse:
        """Deserialise a JSON blob into a ProviderResponse."""
        data = json.loads(blob)
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in data.get("tool_calls", [])
        ]
        return ProviderResponse(
            content=data.get("content"),
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            raw_response=None,
        )
