"""Tests for chat router serialization and session helpers."""

import json
from types import SimpleNamespace
from datetime import datetime, timezone

from apps.api.routers.chat import _serialize_session, _serialize_message, _sse_event


def _make_session(**kwargs):
    defaults = {
        "id": 1,
        "user_id": 10,
        "title": "Test chat",
        "preferences": None,
        "is_active": True,
        "created_at": datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 4, 11, 12, 5, 0, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_message(**kwargs):
    defaults = {
        "id": 42,
        "session_id": 1,
        "role": "assistant",
        "content": "Here are some books.",
        "tool_name": None,
        "tool_input": None,
        "tool_output": None,
        "referenced_book_ids": None,
        "model_used": "test-model",
        "created_at": datetime(2026, 4, 11, 12, 1, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 4, 11, 12, 1, 0, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_serialize_session():
    session = _make_session()
    result = _serialize_session(session)
    assert result["id"] == "1"
    assert result["title"] == "Test chat"
    assert "2026-04-11" in result["createdAt"]


def test_serialize_message_basic():
    msg = _make_message()
    result = _serialize_message(msg)
    assert result["id"] == "42"
    assert result["role"] == "assistant"
    assert result["content"] == "Here are some books."
    assert result["toolTrace"] is None
    assert result["referencedBookIds"] == []
    assert result["modelUsed"] == "test-model"


def test_serialize_message_with_tool_trace():
    msg = _make_message(
        tool_name="search_books",
        tool_input='{"query": "fantasy"}',
        tool_output='{"source": "vector_search", "success": true}',
    )
    result = _serialize_message(msg)
    assert result["toolName"] == "search_books"
    assert result["toolTrace"]["tool"] == "search_books"
    assert result["toolTrace"]["input"] == {"query": "fantasy"}
    assert result["toolTrace"]["source"] == "vector_search"


def test_serialize_message_with_referenced_books():
    msg = _make_message(referenced_book_ids="[1, 2, 3]")
    result = _serialize_message(msg)
    assert result["referencedBookIds"] == [1, 2, 3]


def test_serialize_message_handles_malformed_json():
    msg = _make_message(
        tool_input="not json",
        tool_output="also not json",
        referenced_book_ids="bad",
    )
    result = _serialize_message(msg)
    assert result["toolTrace"] is None  # no tool_name set
    assert result["referencedBookIds"] == []


def test_sse_event_format():
    event_str = _sse_event("token", {"text": "hello"})
    assert event_str.startswith("event: token\n")
    assert "data:" in event_str
    assert event_str.endswith("\n\n")
    data_line = event_str.split("\n")[1]
    payload = json.loads(data_line.split("data: ", 1)[1])
    assert payload["text"] == "hello"
