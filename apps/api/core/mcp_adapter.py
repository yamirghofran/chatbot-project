"""Adapter for the teammates' MCP recommendation server.

Wraps the external MCP service behind a simple Python interface with
automatic timeout handling and graceful degradation.  When the MCP
server is unavailable or returns an error, the adapter returns an empty
list so callers can fall back to the local discovery pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

_log = logging.getLogger(__name__)


class MCPAdapter:
    def __init__(self, base_url: str | None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout

    def is_available(self) -> bool:
        return self._base_url is not None

    def recommend(
        self,
        *,
        user_id: int | None = None,
        preferences: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Request recommendations from the MCP server.

        Returns a list of recommendation dicts on success, or an empty
        list on any failure (timeout, connection error, bad response).
        """
        if self._base_url is None:
            return []

        payload: dict[str, Any] = {"limit": limit}
        if user_id is not None:
            payload["user_id"] = user_id
        if preferences:
            payload["preferences"] = preferences
        if constraints:
            payload["constraints"] = constraints

        try:
            resp = requests.post(
                f"{self._base_url}/recommend",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log.warning("MCP recommendation request failed: %s", e)
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("recommendations", data.get("books", []))
        return []

    def health(self) -> bool:
        """Quick liveness check against the MCP server."""
        if self._base_url is None:
            return False
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=3.0)
            return resp.ok
        except Exception:
            return False
