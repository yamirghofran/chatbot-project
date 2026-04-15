"""Chat orchestrator: tool-routed LLM conversation loop.

Takes a user message + conversation history, decides whether to call a tool,
executes it, and streams a grounded response back via a callback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from groq import APIError, Groq
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from bookdb.models.chatbot_llm import DEFAULT_CHATBOT_MODEL, create_groq_client_sync

from .chat_tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from .config import settings
from .mcp_adapter import MCPAdapter

DEFAULT_ORCHESTRATOR_MODEL = os.environ.get(
    "CHAT_ORCHESTRATOR_MODEL",
    os.environ.get("DEFAULT_CHATBOT_MODEL", "llama-3.3-70b-versatile"),
)

_SYSTEM_PROMPT = """\
You are BookDB Assistant, a knowledgeable and friendly book recommendation assistant \
for BookDB — a social book tracking platform.

You have access to tools for searching books, getting recommendations, comparing books, \
and fetching book details.

Rules:
1. For general book knowledge (authors, series info, plot summaries, publication facts), \
answer directly from your own knowledge. Do NOT call a tool for simple factual questions.
2. Use tools when the user wants personalised recommendations, wants to search the BookDB \
catalogue, needs specific book IDs, or asks to compare books in the database.
3. When you do use a tool, cite specific books from the results. Never invent books that \
aren't in the tool results.
4. Be conversational but concise. Aim for 2-4 sentences per recommendation, not essays.
5. When referencing books from tools, mention them by title so the frontend can render cards.
6. If a tool returns an error or no results, tell the user honestly and answer from your \
own knowledge if possible.
7. For comparison requests, use the compare_books tool.
8. For follow-up refinements like "less romance" or "shorter books", re-search with updated criteria.
9. You can have normal conversations too — greetings, book discussions, reading advice.
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
    extracted_preferences: dict[str, Any] | None = None


StreamCallback = Callable[[str, dict[str, Any]], None]

_PREFERENCE_KEYWORDS = frozenset(
    {
        "genre",
        "recommend",
        "like",
        "dislike",
        "prefer",
        "hate",
        "love",
        "shorter",
        "longer",
        "less",
        "more",
        "avoid",
        "only",
        "standalone",
        "mood",
        "dark",
        "light",
        "funny",
        "serious",
        "fast",
        "slow",
        "romance",
        "fantasy",
        "sci-fi",
        "mystery",
        "horror",
        "thriller",
        "literary",
        "classic",
        "modern",
        "pages",
        "page",
    }
)

_PREFERENCES_EXTRACTION_PROMPT = """\
You are a preference extraction assistant. Given the latest exchange between \
a user and a book recommendation assistant, extract any reading preferences \
the user expressed.

Return a JSON object with ONLY the fields that have evidence in the conversation. \
Omit fields where you have no evidence. Merge with any existing preferences provided.

Fields (all optional):
- liked_genres: list of genres the user likes
- disliked_genres: list of genres the user dislikes
- max_pages: maximum page count (integer)
- standalone_only: true if user wants only standalone books
- preferred_mood: mood the user prefers (e.g. "dark", "light", "funny")
- liked_books: list of book titles the user mentioned positively
- disliked_books: list of book titles the user mentioned negatively
- other_constraints: list of other constraints (e.g. "female protagonist")
"""

_PREFERENCES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "user_preferences",
        "schema": {
            "type": "object",
            "properties": {
                "liked_genres": {"type": "array", "items": {"type": "string"}},
                "disliked_genres": {"type": "array", "items": {"type": "string"}},
                "max_pages": {"type": "integer"},
                "standalone_only": {"type": "boolean"},
                "preferred_mood": {"type": "string"},
                "liked_books": {"type": "array", "items": {"type": "string"}},
                "disliked_books": {"type": "array", "items": {"type": "string"}},
                "other_constraints": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}


def _should_extract_preferences(user_message: str, assistant_content: str) -> bool:
    """Check if the exchange likely contains preference signals."""
    combined = (user_message + " " + assistant_content).lower()
    return any(kw in combined for kw in _PREFERENCE_KEYWORDS)


def _merge_preferences(
    existing: dict[str, Any] | None,
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """Merge newly extracted preferences into existing ones."""
    if not existing:
        return extracted
    merged = dict(existing)
    for key, value in extracted.items():
        if value is None:
            continue
        if isinstance(value, list) and isinstance(merged.get(key), list):
            combined = list(dict.fromkeys(merged[key] + value))
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def _extract_preferences(
    client: Groq,
    model: str,
    user_message: str,
    assistant_content: str,
    existing_preferences: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract structured preferences from the latest exchange."""
    context = ""
    if existing_preferences:
        context = f"\n\nExisting preferences to merge with:\n{json.dumps(existing_preferences)}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PREFERENCES_EXTRACTION_PROMPT + context},
                {
                    "role": "user",
                    "content": f"User said: {user_message}\n\nAssistant replied: {assistant_content[:500]}",
                },
            ],
            response_format=_PREFERENCES_SCHEMA,
            temperature=0.1,
            max_completion_tokens=256,
        )
        raw = response.choices[0].message.content
        if not raw:
            return None
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not parsed:
            return None
        return _merge_preferences(existing_preferences, parsed)
    except Exception:
        return None


def _build_system_message(preferences: dict[str, Any] | None) -> str:
    prompt = _SYSTEM_PROMPT
    if preferences:
        prompt += _PREFERENCES_ADDENDUM.format(preferences=json.dumps(preferences))
    return prompt


def _truncate_history(
    messages: list[dict[str, str]],
    max_messages: int,
) -> list[dict[str, str]]:
    """Keep the last max_messages entries, always preserving the system message."""
    if len(messages) <= max_messages + 1:
        return messages
    return [messages[0]] + messages[-(max_messages):]


def _is_degenerate(text: str, *, threshold: float = 0.4) -> bool:
    """Detect degenerate repetitive output by checking token diversity."""
    if not text or len(text) < 80:
        return False
    tokens = text.split()
    if len(tokens) < 20:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < threshold


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

    # Some models emit function calls as text instead of structured tool_calls.
    # Detect <function=name>{...}</function> or similar patterns and convert them.
    if not tool_calls and content:
        content, text_tool_calls = _extract_text_function_calls(content)
        if text_tool_calls:
            tool_calls = text_tool_calls

    return content, tool_calls


def _message_to_content_and_tool_calls(
    message: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse content and tool calls from a non-streaming assistant message."""
    content = (getattr(message, "content", None) or "") if message else ""
    tool_calls: list[dict[str, Any]] = []
    raw_tcs = getattr(message, "tool_calls", None) if message else None
    if raw_tcs:
        for i, tc in enumerate(raw_tcs):
            fn = getattr(tc, "function", None)
            name = (getattr(fn, "name", None) or "") if fn else ""
            args = (getattr(fn, "arguments", None) or "") if fn else ""
            tool_calls.append(
                {
                    "id": getattr(tc, "id", None) or f"call_{i}",
                    "name": name,
                    "arguments": args,
                }
            )
    if not tool_calls and content:
        content, text_tool_calls = _extract_text_function_calls(content)
        if text_tool_calls:
            tool_calls = text_tool_calls
    return content, tool_calls


def _parse_failed_generation(err: APIError) -> tuple[str, list[dict[str, Any]]]:
    """Try to salvage a tool call from Groq's failed_generation error body."""
    body = getattr(err, "body", None)
    if not body or not isinstance(body, dict):
        return "", []
    failed = body.get("error", {}).get("failed_generation", "")
    if not failed:
        return "", []
    # The failed generation is the raw text the model produced — may contain
    # <tool_call> JSON or <function=...> tags.
    content, tool_calls = _extract_text_function_calls(failed)
    if tool_calls:
        return content, tool_calls
    # Try to parse as raw JSON tool call: {"name": "...", "arguments": {...}}
    try:
        parsed = json.loads(failed)
        if isinstance(parsed, dict) and "name" in parsed:
            args = parsed.get("arguments") or parsed.get("parameters") or {}
            return "", [
                {
                    "id": "recovered_0",
                    "name": parsed["name"],
                    "arguments": json.dumps(args)
                    if not isinstance(args, str)
                    else args,
                }
            ]
    except (json.JSONDecodeError, TypeError):
        pass
    return failed, []


_log = logging.getLogger(__name__)


def _first_turn_with_tools(
    client: Groq,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    freq_penalty: float,
) -> tuple[str, list[dict[str, Any]]]:
    """First model turn with tools: prefer streaming, fall back on Groq tool errors.

    Fallback order:
    1. Streaming + tools
    2. If stream raises APIError, try to parse the failed_generation body
    3. Non-streaming + tools
    4. Non-streaming without tools (last resort)
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "frequency_penalty": freq_penalty,
        "parallel_tool_calls": False,
    }

    # 1. Streaming with tools (preferred path)
    try:
        stream = client.chat.completions.create(stream=True, **kwargs)
        try:
            return _extract_tool_calls_from_stream(stream)
        except APIError as e:
            _log.warning("Groq stream tool error: %s — trying to recover", e)
            content, tool_calls = _parse_failed_generation(e)
            if tool_calls:
                _log.info("Recovered tool call from failed_generation")
                return content, tool_calls
    except APIError as e:
        _log.warning("Groq stream create error: %s — falling back to non-streaming", e)
        content, tool_calls = _parse_failed_generation(e)
        if tool_calls:
            return content, tool_calls

    # 2. Non-streaming with tools
    try:
        _log.info("Retrying non-streaming with tools")
        completion = client.chat.completions.create(stream=False, **kwargs)
        msg = completion.choices[0].message if completion.choices else None
        return _message_to_content_and_tool_calls(msg)
    except APIError as e:
        _log.warning("Groq non-streaming tool error: %s — dropping tools", e)
        content, tool_calls = _parse_failed_generation(e)
        if tool_calls:
            return content, tool_calls

    # 3. Last resort: no tools, plain text response
    _log.warning("All tool attempts failed, completing without tools")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        frequency_penalty=freq_penalty,
        stream=False,
    )
    msg = completion.choices[0].message if completion.choices else None
    content, _ = _message_to_content_and_tool_calls(msg)
    return content, []


_TEXT_FUNC_RE = re.compile(
    r"<function=(\w+)>?(.*?)</function>",
    re.DOTALL,
)


def _extract_text_function_calls(
    content: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse function calls embedded as text and strip them from content."""
    matches = list(_TEXT_FUNC_RE.finditer(content))
    if not matches:
        return content, []

    tool_calls: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        tool_calls.append(
            {
                "id": f"text_call_{i}",
                "name": m.group(1),
                "arguments": m.group(2).strip(),
            }
        )

    cleaned = _TEXT_FUNC_RE.sub("", content).strip()
    return cleaned, tool_calls


_MIN_SEARCH_QUERY_WORDS = 3


def _enrich_search_query(
    raw_query: str,
    history: list[dict[str, Any]],
) -> str:
    """If the LLM passed a short/vague query, expand it from conversation context.

    Scans recent history for substantive user messages that provide topical
    context (e.g. "I just finished Harry Potter and loved it") and prepends
    that context to the raw query so the embedding has real signal.
    """
    words = raw_query.split()
    if len(words) >= _MIN_SEARCH_QUERY_WORDS:
        return raw_query

    context_parts: list[str] = []
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        text = (msg.get("content") or "").strip()
        if len(text.split()) >= _MIN_SEARCH_QUERY_WORDS:
            context_parts.append(text)
            if len(context_parts) >= 2:
                break

    if not context_parts:
        return raw_query

    context_parts.reverse()
    enriched = "; ".join(context_parts) + " — " + raw_query
    return enriched


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
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatch a tool call to the correct function."""
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "books": [],
            "data": {},
            "source": "",
        }

    if tool_name == "search_books":
        query = _enrich_search_query(tool_args.get("query", ""), history or [])
        sentiments_df = getattr(request_app_state, "book_sentiments_df", None) if request_app_state else None
        return func(
            query,
            db=db,
            qdrant=qdrant,
            groq_client=groq_client,
            preferences=preferences,
            sentiments_df=sentiments_df,
        )
    elif tool_name == "get_book_details":
        return func(tool_args.get("book_id", 0), db=db)
    elif tool_name == "get_related_books":
        return func(
            tool_args.get("book_id", 0),
            db=db,
            qdrant=qdrant,
            limit=tool_args.get("limit", 6),
        )
    elif tool_name == "get_recommendations":
        return func(
            db=db,
            qdrant=qdrant,
            request_app_state=request_app_state,
            user_id=user_id,
            limit=tool_args.get("limit", 6),
        )
    elif tool_name == "compare_books":
        return func(
            book_ids=tool_args.get("book_ids"),
            titles=tool_args.get("titles"),
            db=db,
            groq_client=groq_client,
        )
    elif tool_name == "recommend_via_mcp":
        return func(
            mcp_adapter=mcp_adapter,
            db=db,
            qdrant=qdrant,
            request_app_state=request_app_state,
            user_id=user_id,
            preferences=preferences,
            limit=tool_args.get("limit", 6),
        )
    else:
        return {
            "success": False,
            "error": f"Unhandled tool: {tool_name}",
            "books": [],
            "data": {},
            "source": "",
        }


def _collect_book_ids(books: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for b in books:
        try:
            ids.append(int(b["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def _has_meaningful_output(output: dict[str, Any]) -> bool:
    """True if a tool result is successful and has books or non-empty structured data."""
    if not output.get("success", False):
        return False
    if output.get("books"):
        return True
    data = output.get("data")
    if isinstance(data, dict):
        return any(v not in (None, "", [], {}, ()) for v in data.values())
    return bool(data)


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

    model = DEFAULT_ORCHESTRATOR_MODEL or DEFAULT_CHATBOT_MODEL
    client = groq_client or create_groq_client_sync()
    cb = stream_callback or (lambda _evt, _data: None)

    system_msg = _build_system_message(preferences)
    max_history = settings.CHAT_MAX_HISTORY_MESSAGES

    # Filter degenerate messages from history so bad outputs don't poison context
    clean_history = [
        m for m in history if m.get("content") and not _is_degenerate(m["content"])
    ]

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_msg}]
    messages.extend(clean_history)
    messages.append({"role": "user", "content": user_message})
    messages = _truncate_history(messages, max_history)

    temperature = float(os.environ.get("CHATBOT_TEMPERATURE", "0.6"))
    max_tokens = int(os.environ.get("MAX_CHATBOT_TOKENS", "1024"))
    freq_penalty = float(os.environ.get("CHATBOT_FREQUENCY_PENALTY", "0.3"))

    # First LLM call — may produce a tool call or a direct response.
    # Groq sometimes raises APIError mid-stream ("Failed to call a function");
    # _first_turn_with_tools retries buffered then tool-less completion.
    try:
        content, tool_calls = _first_turn_with_tools(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            freq_penalty=freq_penalty,
        )
    except Exception as e:
        error_msg = f"I'm having trouble connecting to the AI service right now. Please try again shortly. ({e})"
        cb("token", {"text": error_msg})
        cb("done", {"message_id": "", "referenced_book_ids": [], "model_used": model})
        return OrchestratorResult(content=error_msg, model_used=model)

    # If no tool calls, the model responded directly — stream what we got
    if not tool_calls:
        if content:
            cb("token", {"text": content})
        prefs_out = None
        if content and _should_extract_preferences(user_message, content):
            prefs_out = _extract_preferences(
                client, model, user_message, content, preferences
            )
        cb("done", {"message_id": "", "referenced_book_ids": [], "model_used": model})
        return OrchestratorResult(
            content=content, model_used=model, extracted_preferences=prefs_out
        )

    # Execute tool calls
    all_books: list[dict[str, Any]] = []
    all_tool_records: list[ToolCallRecord] = []

    for tc in tool_calls:
        tool_name = tc["name"]
        try:
            raw_args = tc.get("arguments") or ""
            tool_args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            tool_args = {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        cb("tool_call", {"tool": tool_name, "input": tool_args})

        tool_result = _execute_tool(
            tool_name,
            tool_args,
            db=db,
            qdrant=qdrant_client,
            groq_client=client,
            mcp_adapter=mcp_adapter,
            request_app_state=request_app_state,
            user_id=user_id,
            preferences=preferences,
            history=messages,
        )

        tool_books = tool_result.get("books", [])
        all_books.extend(tool_books)
        all_tool_records.append(
            ToolCallRecord(
                name=tool_name,
                input=tool_args,
                output=tool_result,
            )
        )

        cb(
            "tool_result",
            {
                "tool": tool_name,
                "books": tool_books,
                "source": tool_result.get("source", ""),
                "data": tool_result.get("data", {}),
            },
        )

        comparison = tool_result.get("data", {}).get("comparison")
        if comparison:
            cb(
                "comparison",
                {
                    "dimensions": comparison.get("dimensions", []),
                    "verdict": comparison.get("verdict", ""),
                    "book_ids": _collect_book_ids(tool_books),
                },
            )

    # Check if ALL tools failed or returned no books
    all_failed = all(not _has_meaningful_output(rec.output) for rec in all_tool_records)

    # If every tool failed/empty, fall back to a direct answer without tool context
    if all_failed:
        fallback_msgs = list(messages)
        fallback_msgs.append(
            {
                "role": "system",
                "content": (
                    "The database tools returned no results (the catalogue may be "
                    "empty or the search service is unavailable). Answer the user's "
                    "question from your own knowledge. Be honest that you couldn't "
                    "search the BookDB catalogue."
                ),
            }
        )
        try:
            fallback_stream = client.chat.completions.create(
                model=model,
                messages=fallback_msgs,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                frequency_penalty=freq_penalty,
                stream=True,
            )
            parts: list[str] = []
            for chunk in fallback_stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.delta and choice.delta.content:
                    text = choice.delta.content
                    parts.append(text)
                    cb("token", {"text": text})
            fallback_content = "".join(parts)
            fb_ids = _collect_book_ids(all_books)
            cb("done", {"message_id": "", "referenced_book_ids": fb_ids, "model_used": model})
            return OrchestratorResult(
                content=fallback_content,
                referenced_book_ids=fb_ids,
                tool_calls=all_tool_records,
                books=all_books,
                model_used=model,
            )
        except Exception:
            msg = "I wasn't able to search the book catalogue right now. Could you try again?"
            cb("token", {"text": msg})
            cb("done", {"message_id": "", "referenced_book_ids": [], "model_used": model})
            return OrchestratorResult(content=msg, model_used=model)

    # Build tool result message for the second LLM call
    tool_context_parts: list[str] = []
    for record in all_tool_records:
        result = record.output
        books_summary = ""
        for b in result.get("books", []):
            books_summary += f"  - [id:{b.get('id')}] {b.get('title', '?')} by {b.get('author', '?')}\n"

        no_books_line = "  (none)\n"
        tool_context_parts.append(
            f"Tool: {record.name}\n"
            f"Source: {result.get('source', 'unknown')}\n"
            f"Books found:\n{books_summary or no_books_line}"
        )

        comparison = result.get("data", {}).get("comparison")
        if comparison:
            tool_context_parts.append(f"Comparison data: {json.dumps(comparison)}")

    tool_message_content = "\n\n".join(tool_context_parts)

    # Second LLM call — generate grounded response
    follow_up_messages = list(messages)
    follow_up_messages.append(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
    )
    for i, tc in enumerate(tool_calls):
        follow_up_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"] or f"call_{i}",
                "content": tool_message_content if i == 0 else "(see above)",
            }
        )

    try:
        follow_stream = client.chat.completions.create(
            model=model,
            messages=follow_up_messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            frequency_penalty=freq_penalty,
            stream=True,
        )
    except Exception:
        fallback = "I found some books but had trouble generating a summary. Here are the results."
        cb("token", {"text": fallback})
        book_ids = _collect_book_ids(all_books)
        cb("done", {"message_id": "", "referenced_book_ids": book_ids, "model_used": model})
        return OrchestratorResult(
            content=fallback,
            referenced_book_ids=book_ids,
            tool_calls=all_tool_records,
            books=all_books,
            model_used=model,
        )

    final_parts: list[str] = []
    for chunk in follow_stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice and choice.delta and choice.delta.content:
            text = choice.delta.content
            final_parts.append(text)
            cb("token", {"text": text})

    final_content = "".join(final_parts)

    # Safety net: if the model produced degenerate output, replace it
    if _is_degenerate(final_content):
        final_content = (
            "I found some books but had trouble generating a summary. "
            "Here are the results I found."
        )

    book_ids = _collect_book_ids(all_books)

    cb("book_cards", {"books": all_books})

    prefs_out = None
    if _should_extract_preferences(user_message, final_content):
        prefs_out = _extract_preferences(
            client, model, user_message, final_content, preferences
        )

    cb("done", {"message_id": "", "referenced_book_ids": book_ids, "model_used": model})
    return OrchestratorResult(
        content=final_content,
        referenced_book_ids=book_ids,
        tool_calls=all_tool_records,
        books=all_books,
        model_used=model,
        extracted_preferences=prefs_out,
    )
