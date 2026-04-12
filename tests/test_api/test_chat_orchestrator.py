"""Tests for the chat orchestrator."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from apps.api.core import chat_orchestrator, chat_tools


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    """Provide minimal settings so the orchestrator doesn't need real env vars."""
    fake_settings = SimpleNamespace(CHAT_MAX_HISTORY_MESSAGES=10)
    monkeypatch.setattr(chat_orchestrator, "settings", fake_settings)


def _mock_groq_stream_direct(content: str):
    """Create a mock Groq streaming response that returns content directly."""
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
            )
        ]
    )
    return [chunk]


def _mock_groq_stream_tool_call(tool_name: str, arguments: str):
    """Create a mock Groq streaming response with a tool call."""
    tc = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name=tool_name, arguments=arguments),
    )
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc]),
            )
        ]
    )
    return [chunk]


def test_orchestrate_direct_response():
    """When the LLM responds without a tool call, content is returned directly."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_groq_stream_direct("Hello! How can I help?")

    events = []
    result = chat_orchestrator.orchestrate(
        user_message="Hi",
        history=[],
        db=MagicMock(),
        groq_client=client,
        stream_callback=lambda evt, data: events.append((evt, data)),
    )

    assert result.content == "Hello! How can I help?"
    assert result.tool_calls == []
    assert any(evt == "token" for evt, _ in events)
    assert any(evt == "done" for evt, _ in events)


def test_orchestrate_tool_call_flow():
    """When the LLM requests a tool call, the tool is executed and a follow-up response is generated."""
    client = MagicMock()

    # First call returns a tool call, second returns content
    client.chat.completions.create.side_effect = [
        _mock_groq_stream_tool_call("search_books", '{"query": "fantasy books"}'),
        _mock_groq_stream_direct("Based on my search, I recommend..."),
    ]

    mock_tool_result = {
        "success": True,
        "data": {"llm_context_books": [], "reviews": [], "book_count": 2},
        "books": [
            {"id": "1", "title": "Book One", "author": "Author A"},
            {"id": "2", "title": "Book Two", "author": "Author B"},
        ],
        "source": "vector_search",
    }

    events = []

    with patch.dict(chat_tools.TOOL_FUNCTIONS, {"search_books": lambda *a, **kw: mock_tool_result}):
        result = chat_orchestrator.orchestrate(
            user_message="recommend fantasy books",
            history=[],
            db=MagicMock(),
            groq_client=client,
            stream_callback=lambda evt, data: events.append((evt, data)),
        )

    assert "recommend" in result.content.lower() or "search" in result.content.lower() or len(result.content) > 0
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_books"
    assert len(result.books) == 2

    event_types = [evt for evt, _ in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types


def test_orchestrate_handles_llm_error():
    """When the LLM fails, a graceful error message is returned."""
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API unavailable")

    events = []
    result = chat_orchestrator.orchestrate(
        user_message="hello",
        history=[],
        db=MagicMock(),
        groq_client=client,
        stream_callback=lambda evt, data: events.append((evt, data)),
    )

    assert "trouble" in result.content.lower() or "error" in result.content.lower()
    assert any(evt == "done" for evt, _ in events)


def test_orchestrate_all_tools_fail_fallback():
    """When every tool returns failure/empty, the orchestrator falls back to a direct answer."""
    client = MagicMock()

    # First call triggers a tool, fallback call returns direct content
    client.chat.completions.create.side_effect = [
        _mock_groq_stream_tool_call("search_books", '{"query": "harry potter"}'),
        _mock_groq_stream_direct("The Harry Potter series has 7 books."),
    ]

    empty_result = {
        "success": False,
        "data": {},
        "books": [],
        "source": "",
        "error": "Embedding failed",
    }

    events = []

    with patch.dict(chat_tools.TOOL_FUNCTIONS, {"search_books": lambda *a, **kw: empty_result}):
        result = chat_orchestrator.orchestrate(
            user_message="How many Harry Potter books are there?",
            history=[],
            db=MagicMock(),
            groq_client=client,
            stream_callback=lambda evt, data: events.append((evt, data)),
        )

    assert "7" in result.content or "Harry Potter" in result.content
    assert any(evt == "done" for evt, _ in events)


def test_degenerate_detection():
    """Degenerate repetitive text is detected correctly."""
    normal = "The Harry Potter series has seven books written by J.K. Rowling."
    assert not chat_orchestrator._is_degenerate(normal)

    garbage = " ".join(["Mr Mc France Paris sn blue"] * 30)
    assert chat_orchestrator._is_degenerate(garbage)

    short = "Mr Mr Mr"
    assert not chat_orchestrator._is_degenerate(short)


def test_degenerate_history_filtered():
    """Degenerate assistant messages are stripped from history before sending to LLM."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_groq_stream_direct("Seven books total.")

    garbage_history = [
        {"role": "user", "content": "How many HP books?"},
        {"role": "assistant", "content": " ".join(["Mr Mc France Paris sn blue"] * 30)},
    ]

    result = chat_orchestrator.orchestrate(
        user_message="How many books?",
        history=garbage_history,
        db=MagicMock(),
        groq_client=client,
        stream_callback=lambda _e, _d: None,
    )

    # The call should succeed and the degenerate message should not be in the messages
    call_args = client.chat.completions.create.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
    contents = [m.get("content", "") for m in sent_messages]
    assert not any(chat_orchestrator._is_degenerate(c) for c in contents if c)
    assert result.content == "Seven books total."


def test_text_function_call_parsed():
    """Function calls emitted as text are parsed and executed as tool calls."""
    client = MagicMock()

    text_with_func = (
        'It seems like you want sci-fi. '
        '<function=get_recommendations>{"genre":"science fiction","limit":6}</function>'
    )
    # First call returns function-as-text, second returns grounded response
    client.chat.completions.create.side_effect = [
        _mock_groq_stream_direct(text_with_func),
        _mock_groq_stream_direct("Here are some great sci-fi picks!"),
    ]

    mock_result = {
        "success": True,
        "data": {},
        "books": [{"id": "1", "title": "Dune", "author": "Frank Herbert"}],
        "source": "cold_start",
    }

    events = []

    with patch.dict(chat_tools.TOOL_FUNCTIONS, {"get_recommendations": lambda *a, **kw: mock_result}):
        result = chat_orchestrator.orchestrate(
            user_message="I like sci-fi books",
            history=[],
            db=MagicMock(),
            groq_client=client,
            stream_callback=lambda evt, data: events.append((evt, data)),
        )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_recommendations"
    event_types = [evt for evt, _ in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types


def test_text_function_call_stripped_from_content():
    """The raw function tag is removed from content shown to the user."""
    content = 'Let me search. <function=search_books>{"query":"fantasy"}</function>'
    cleaned, calls = chat_orchestrator._extract_text_function_calls(content)
    assert "<function" not in cleaned
    assert len(calls) == 1
    assert calls[0]["name"] == "search_books"
    assert "fantasy" in calls[0]["arguments"]


def test_enrich_search_query_short_input():
    """Short queries get enriched with prior user messages from history."""
    history = [
        {"role": "user", "content": "I just finished reading Harry Potter and loved the magic"},
        {"role": "assistant", "content": "Great choice! Want recommendations?"},
        {"role": "user", "content": "yes"},
    ]
    enriched = chat_orchestrator._enrich_search_query("yes", history)
    assert "Harry Potter" in enriched
    assert "yes" in enriched


def test_enrich_search_query_already_descriptive():
    """Queries that are already descriptive pass through unchanged."""
    history = [
        {"role": "user", "content": "something about wizards"},
    ]
    long_query = "dark fantasy books with complex magic systems"
    result = chat_orchestrator._enrich_search_query(long_query, history)
    assert result == long_query


def test_enrich_search_query_no_history():
    """Short queries without usable history pass through unchanged."""
    result = chat_orchestrator._enrich_search_query("yes", [])
    assert result == "yes"


def test_truncate_history():
    """History is truncated to max_messages while preserving the system message."""
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "resp2"},
        {"role": "user", "content": "msg3"},
    ]

    result = chat_orchestrator._truncate_history(messages, max_messages=3)
    assert result[0]["role"] == "system"
    assert len(result) == 4  # system + last 3
    assert result[-1]["content"] == "msg3"
