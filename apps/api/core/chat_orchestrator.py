"""Chat orchestrator: tool-routed LLM conversation loop.

Takes a user message + conversation history, decides whether to call a tool,
executes it, and streams a grounded response back via a callback.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from groq import Groq
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from .chat_tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from .config import settings
from .mcp_adapter import MCPAdapter

DEFAULT_ORCHESTRATOR_MODEL = os.environ.get(
    "CHAT_ORCHESTRATOR_MODEL",
    os.environ.get("DEFAULT_CHATBOT_MODEL", "moonshotai/kimi-k2-instruct-0905"),
)

_SYSTEM_PROMPT = """\
You are BookDB Assistant, a knowledgeable and friendly book recommendation assistant \
for BookDB — a social book tracking platform.

You have access to tools for searching books, getting recommendations, comparing books, \
and fetching book details. Use them when the user asks about books.

Rules:
1. Always use a tool when the user asks for book suggestions, information, or comparisons.
2. Cite specific books from tool results. Never invent books that aren't in the results.
3. Be conversational but concise. Aim for 2-4 sentences per recommendation, not essays.
4. When referencing books, always mention them by title so the frontend can render book cards.
5. If you're unsure or lack data, say so honestly rather than guessing.
6. For comparison requests, use the compare_books tool.
7. For follow-up refinements like "less romance" or "shorter books", re-search with updated criteria.
8. You can have normal conversations too — greetings, book discussions, reading advice.
9. When multiple tools could help, prefer the most specific one.
"""

_PREFERENCES_ADDENDUM = """
Current user preferences from this conversation:
{preferences}
"""


@dataclass
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class OrchestratorResult:
    content: str
    referenced_book_ids: list[int] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    books: list[dict[str, Any]] = field(default_factory=list)
    model_used: str = ""


StreamCallback = Callable[[str, dict[str, Any]], None]


def _build_system_message(preferences: dict[str, Any] | None) -> str:
    prompt = _SYSTEM_PROMPT
    if preferences:
        prompt += _PREFERENCES_ADDENDUM.format(preferences=json.dumps(preferences))
    return prompt


def _truncate_history(
    messages: list[dict[str, str]], max_messages: int,
) -> list[dict[str, str]]:
    """Keep the last max_messages entries, always preserving the system message."""
    if len(messages) <= max_messages + 1:
        return messages
    return [messages[0]] + messages[-(max_messages):]


def _extract_tool_calls_from_stream(stream) -> tuple[str, list[dict[str, Any]]]:
    """Consume a Groq streaming response, accumulating content and tool calls."""
    content_parts: list[str] = []
    tool_calls_accum: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue
        delta = choice.delta

        if delta.content:
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_accum:
                    tool_calls_accum[idx] = {
                        "id": tc.id or "",
                        "name": "",
                        "arguments": "",
                    }
                if tc.function:
                    if tc.function.name:
                        tool_calls_accum[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_accum[idx]["arguments"] += tc.function.arguments

    content = "".join(content_parts)
    tool_calls = [tool_calls_accum[k] for k in sorted(tool_calls_accum)]
    return content, tool_calls


def _execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    db: Session,
    qdrant: QdrantClient | None,
    groq_client: Groq,
    mcp_adapter: MCPAdapter | None,
    request_app_state: Any | None,
    user_id: int | None,
    preferences: dict[str, Any] | None,
) -> dict[str, Any]:
    """Dispatch a tool call to the correct function."""
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return {"success": False, "error": f"Unknown tool: {tool_name}", "books": [], "data": {}, "source": ""}

    if tool_name == "search_books":
        return func(tool_args.get("query", ""), db=db, qdrant=qdrant, groq_client=groq_client)
    elif tool_name == "get_book_details":
        return func(tool_args.get("book_id", 0), db=db)
    elif tool_name == "get_related_books":
        return func(tool_args.get("book_id", 0), db=db, qdrant=qdrant, limit=tool_args.get("limit", 6))
    elif tool_name == "get_recommendations":
        return func(
            db=db, qdrant=qdrant, request_app_state=request_app_state,
            user_id=user_id, limit=tool_args.get("limit", 6),
        )
    elif tool_name == "compare_books":
        return func(tool_args.get("book_ids", []), db=db, groq_client=groq_client)
    elif tool_name == "recommend_via_mcp":
        return func(
            mcp_adapter=mcp_adapter, db=db, qdrant=qdrant,
            request_app_state=request_app_state, user_id=user_id,
            preferences=preferences, limit=tool_args.get("limit", 6),
        )
    else:
        return {"success": False, "error": f"Unhandled tool: {tool_name}", "books": [], "data": {}, "source": ""}


def _collect_book_ids(books: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for b in books:
        try:
            ids.append(int(b["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def orchestrate(
    *,
    user_message: str,
    history: list[dict[str, str]],
    preferences: dict[str, Any] | None = None,
    db: Session,
    qdrant_client: QdrantClient | None = None,
    mcp_adapter: MCPAdapter | None = None,
    groq_client: Groq | None = None,
    request_app_state: Any | None = None,
    user_id: int | None = None,
    stream_callback: StreamCallback | None = None,
) -> OrchestratorResult:
    """Run one orchestration turn.

    Args:
        user_message: The user's new message.
        history: Prior messages as ``[{role, content}, ...]``.
        preferences: Accumulated session preferences (nullable).
        db: SQLAlchemy session.
        qdrant_client: Qdrant client (optional).
        mcp_adapter: MCP adapter (optional).
        groq_client: Groq client (created if not provided).
        request_app_state: FastAPI app.state for accessing BPR/metrics paths.
        user_id: Authenticated user ID (optional).
        stream_callback: ``(event_type, data_dict)`` called for each SSE event.

    Returns:
        OrchestratorResult with the final content, book IDs, and tool traces.
    """
    from bookdb.models.chatbot_llm import DEFAULT_CHATBOT_MODEL, create_groq_client_sync

    model = DEFAULT_ORCHESTRATOR_MODEL or DEFAULT_CHATBOT_MODEL
    client = groq_client or create_groq_client_sync()
    cb = stream_callback or (lambda _evt, _data: None)

    system_msg = _build_system_message(preferences)
    max_history = settings.CHAT_MAX_HISTORY_MESSAGES

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    messages = _truncate_history(messages, max_history)

    # First LLM call — may produce a tool call or a direct response
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            temperature=float(os.environ.get("CHATBOT_TEMPERATURE", "0.7")),
            max_completion_tokens=int(os.environ.get("MAX_CHATBOT_TOKENS", "1024")),
            stream=True,
        )
    except Exception as e:
        error_msg = f"I'm having trouble connecting to the AI service right now. Please try again shortly. ({e})"
        cb("token", {"text": error_msg})
        cb("done", {"referenced_book_ids": []})
        return OrchestratorResult(content=error_msg, model_used=model)

    content, tool_calls = _extract_tool_calls_from_stream(stream)

    # If no tool calls, the model responded directly — stream what we got
    if not tool_calls:
        if content:
            cb("token", {"text": content})
        cb("done", {"referenced_book_ids": []})
        return OrchestratorResult(content=content, model_used=model)

    # Execute tool calls
    all_books: list[dict[str, Any]] = []
    all_tool_records: list[ToolCallRecord] = []

    for tc in tool_calls:
        tool_name = tc["name"]
        try:
            tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            tool_args = {}

        cb("tool_call", {"tool": tool_name, "input": tool_args})

        tool_result = _execute_tool(
            tool_name, tool_args,
            db=db, qdrant=qdrant_client, groq_client=client,
            mcp_adapter=mcp_adapter, request_app_state=request_app_state,
            user_id=user_id, preferences=preferences,
        )

        tool_books = tool_result.get("books", [])
        all_books.extend(tool_books)
        all_tool_records.append(ToolCallRecord(
            name=tool_name, input=tool_args, output=tool_result,
        ))

        cb("tool_result", {
            "tool": tool_name,
            "books": tool_books,
            "source": tool_result.get("source", ""),
        })

    # Build tool result message for the second LLM call
    tool_context_parts: list[str] = []
    for record in all_tool_records:
        result = record.output
        books_summary = ""
        for b in result.get("books", []):
            books_summary += f"  - [id:{b.get('id')}] {b.get('title', '?')} by {b.get('author', '?')}\n"

        tool_context_parts.append(
            f"Tool: {record.name}\n"
            f"Source: {result.get('source', 'unknown')}\n"
            f"Books found:\n{books_summary or '  (none)\n'}"
        )

        comparison = result.get("data", {}).get("comparison")
        if comparison:
            tool_context_parts.append(f"Comparison data: {json.dumps(comparison)}")

    tool_message_content = "\n\n".join(tool_context_parts)

    # Second LLM call — generate grounded response
    follow_up_messages = list(messages)
    follow_up_messages.append({"role": "assistant", "content": content, "tool_calls": [
        {"id": tc["id"] or f"call_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
        for i, tc in enumerate(tool_calls)
    ]})
    for i, tc in enumerate(tool_calls):
        follow_up_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"] or f"call_{i}",
            "content": tool_message_content if i == 0 else "(see above)",
        })

    try:
        follow_stream = client.chat.completions.create(
            model=model,
            messages=follow_up_messages,
            temperature=float(os.environ.get("CHATBOT_TEMPERATURE", "0.7")),
            max_completion_tokens=int(os.environ.get("MAX_CHATBOT_TOKENS", "1024")),
            stream=True,
        )
    except Exception as e:
        fallback = "I found some books but had trouble generating a summary. Here are the results."
        cb("token", {"text": fallback})
        book_ids = _collect_book_ids(all_books)
        cb("done", {"referenced_book_ids": book_ids})
        return OrchestratorResult(
            content=fallback, referenced_book_ids=book_ids,
            tool_calls=all_tool_records, books=all_books, model_used=model,
        )

    final_parts: list[str] = []
    for chunk in follow_stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice and choice.delta and choice.delta.content:
            text = choice.delta.content
            final_parts.append(text)
            cb("token", {"text": text})

    final_content = "".join(final_parts)
    book_ids = _collect_book_ids(all_books)

    cb("book_cards", {"books": all_books})
    cb("done", {"referenced_book_ids": book_ids})

    return OrchestratorResult(
        content=final_content,
        referenced_book_ids=book_ids,
        tool_calls=all_tool_records,
        books=all_books,
        model_used=model,
    )
