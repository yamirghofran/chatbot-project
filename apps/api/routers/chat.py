"""Chat session and message endpoints with SSE streaming."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from bookdb.db.chat_models import ChatMessage, ChatSession
from bookdb.db.models import Book, User
from bookdb.models.chatbot_llm import create_groq_client_sync

from ..core.serialize import serialize_book

from ..core.chat_orchestrator import orchestrate
from ..core.config import settings
from ..core.deps import get_db, get_optional_user
from ..core.mcp_adapter import MCPAdapter
from ..schemas.chat import (
    CreateSessionRequest,
    MessageOut,
    SendMessageRequest,
    SessionDetailOut,
    SessionOut,
    ToolTraceOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_session(session: ChatSession) -> dict[str, Any]:
    return {
        "id": str(session.id),
        "title": session.title,
        "createdAt": session.created_at.isoformat() if session.created_at else "",
        "updatedAt": session.updated_at.isoformat() if session.updated_at else "",
    }


def _resolve_books(db: Session, book_ids: list[int]) -> list[dict[str, Any]]:
    """Fetch Book records by ID and return serialized dicts."""
    if not book_ids:
        return []
    books = db.scalars(select(Book).where(Book.id.in_(book_ids))).all()
    book_map = {b.id: b for b in books}
    return [serialize_book(book_map[bid]) for bid in book_ids if bid in book_map]


def _serialize_message(
    msg: ChatMessage, *, books_by_id: dict[int, dict[str, Any]] | None = None
) -> dict[str, Any]:
    tool_traces: list[dict[str, Any]] = []
    comparison_data: dict[str, Any] | None = None

    if msg.tool_name:
        tool_input = None
        tool_output = None
        try:
            tool_input = json.loads(msg.tool_input) if msg.tool_input else None
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            tool_output = json.loads(msg.tool_output) if msg.tool_output else None
        except (json.JSONDecodeError, TypeError):
            pass

        # Multi-tool format: list of {tool, input/output}
        if isinstance(tool_output, list):
            for i, entry in enumerate(tool_output):
                inp = (
                    tool_input[i]["input"]
                    if isinstance(tool_input, list) and i < len(tool_input)
                    else tool_input
                )
                out = entry.get("output", {}) if isinstance(entry, dict) else {}
                trace = {
                    "tool": entry.get("tool", msg.tool_name)
                    if isinstance(entry, dict)
                    else msg.tool_name,
                    "input": inp,
                    "output": out,
                    "source": out.get("source") if isinstance(out, dict) else None,
                }
                tool_traces.append(trace)
                comp = (
                    out.get("data", {}).get("comparison")
                    if isinstance(out, dict)
                    else None
                )
                if comp and comparison_data is None:
                    comparison_data = comp
        else:
            trace = {
                "tool": msg.tool_name,
                "input": tool_input,
                "output": tool_output,
                "source": None,
            }
            if isinstance(tool_output, dict):
                trace["source"] = tool_output.get("source")
                comp = tool_output.get("data", {}).get("comparison")
                if comp:
                    comparison_data = comp
            tool_traces.append(trace)

    ref_ids: list[int] = []
    if msg.referenced_book_ids:
        try:
            ref_ids = json.loads(msg.referenced_book_ids)
        except (json.JSONDecodeError, TypeError):
            pass

    # Resolve book objects for persisted messages
    ref_books: list[dict[str, Any]] = []
    if ref_ids and books_by_id:
        ref_books = [books_by_id[bid] for bid in ref_ids if bid in books_by_id]

    primary_trace = tool_traces[0] if tool_traces else None

    result: dict[str, Any] = {
        "id": str(msg.id),
        "role": msg.role,
        "content": msg.content,
        "toolName": msg.tool_name,
        "toolTrace": primary_trace,
        "toolTraces": tool_traces if len(tool_traces) > 1 else None,
        "comparison": comparison_data,
        "referencedBookIds": ref_ids,
        "modelUsed": msg.model_used,
        "timestamp": msg.created_at.isoformat() if msg.created_at else "",
    }
    if ref_books:
        result["referencedBooks"] = ref_books
    return result


def _get_session_or_404(
    db: Session,
    session_id: int,
    user: User | None,
) -> ChatSession:
    session = db.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.is_active.is_(True),
        )
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.user_id is not None:
        if user is None or session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not your session"
            )
    return session


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/sessions", response_model=SessionOut)
def create_session(
    body: CreateSessionRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    session = ChatSession(
        user_id=current_user.id if current_user else None,
        title=body.title or "New conversation",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _serialize_session(session)


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    if current_user is None:
        return []

    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id, ChatSession.is_active.is_(True))
        .order_by(ChatSession.updated_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return [_serialize_session(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    session = _get_session_or_404(db, session_id, current_user)
    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
    ).all()

    # Bulk-resolve all referenced book IDs across messages
    all_book_ids: set[int] = set()
    for m in messages:
        if m.referenced_book_ids:
            try:
                ids = json.loads(m.referenced_book_ids)
                all_book_ids.update(ids)
            except (json.JSONDecodeError, TypeError):
                pass
    books_by_id: dict[int, dict[str, Any]] = {}
    if all_book_ids:
        resolved = _resolve_books(db, list(all_book_ids))
        books_by_id = {int(b["id"]): b for b in resolved}

    prefs = None
    if session.preferences:
        try:
            prefs = json.loads(session.preferences)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "id": str(session.id),
        "title": session.title,
        "messages": [_serialize_message(m, books_by_id=books_by_id) for m in messages],
        "preferences": prefs,
        "createdAt": session.created_at.isoformat() if session.created_at else "",
    }


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    session = _get_session_or_404(db, session_id, current_user)
    session.is_active = False
    db.commit()


@router.patch("/sessions/{session_id}/preferences", status_code=status.HTTP_200_OK)
def update_preferences(
    session_id: int,
    body: dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    session = _get_session_or_404(db, session_id, current_user)
    prefs = body.get("preferences")
    session.preferences = json.dumps(prefs) if prefs else None
    db.commit()
    return {"ok": True}


@router.post("/sessions/{session_id}/messages")
def send_message(
    session_id: int,
    body: SendMessageRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    session = _get_session_or_404(db, session_id, current_user)

    # Persist user message
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=body.content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Load history for context
    max_history = settings.CHAT_MAX_HISTORY_MESSAGES
    history_rows = db.scalars(
        select(ChatMessage)
        .where(
            ChatMessage.session_id == session.id,
            ChatMessage.role.in_(["user", "assistant"]),
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(max_history + 1)
    ).all()
    history_rows = list(reversed(history_rows))
    # Exclude the message we just added (it's passed separately)
    history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
        if m.id != user_msg.id
    ]

    # Session preferences
    prefs = None
    if session.preferences:
        try:
            prefs = json.loads(session.preferences)
        except (json.JSONDecodeError, TypeError):
            pass

    # Resolve dependencies
    qdrant = getattr(request.app.state, "qdrant", None)
    mcp: MCPAdapter | None = getattr(request.app.state, "mcp_adapter", None)
    groq_client = create_groq_client_sync()

    def event_stream():
        events: list[tuple[str, dict[str, Any]]] = []

        def collect_event(event_type: str, data: dict[str, Any]):
            events.append((event_type, data))

        result = orchestrate(
            user_message=body.content,
            history=history,
            preferences=prefs,
            db=db,
            qdrant_client=qdrant,
            mcp_adapter=mcp,
            groq_client=groq_client,
            request_app_state=request.app.state,
            user_id=current_user.id if current_user else None,
            stream_callback=collect_event,
        )

        # Persist assistant message — store all tool calls, not just first
        tool_name = None
        tool_input_json = None
        tool_output_json = None
        if result.tool_calls:
            tool_name = result.tool_calls[0].name
            if len(result.tool_calls) == 1:
                tc = result.tool_calls[0]
                try:
                    tool_input_json = json.dumps(tc.input)
                except (TypeError, ValueError):
                    pass
                try:
                    tool_output_json = json.dumps(tc.output, default=str)
                except (TypeError, ValueError):
                    pass
            else:
                all_inputs = [
                    {"tool": tc.name, "input": tc.input} for tc in result.tool_calls
                ]
                all_outputs = [
                    {"tool": tc.name, "output": tc.output} for tc in result.tool_calls
                ]
                try:
                    tool_input_json = json.dumps(all_inputs)
                except (TypeError, ValueError):
                    pass
                try:
                    tool_output_json = json.dumps(all_outputs, default=str)
                except (TypeError, ValueError):
                    pass

        ref_ids_json = (
            json.dumps(result.referenced_book_ids)
            if result.referenced_book_ids
            else None
        )

        assistant_msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=result.content,
            tool_name=tool_name,
            tool_input=tool_input_json,
            tool_output=tool_output_json,
            referenced_book_ids=ref_ids_json,
            model_used=result.model_used,
        )
        db.add(assistant_msg)

        # Auto-title: use the first user message as session title
        if session.title == "New conversation" and body.content:
            session.title = body.content[:100]

        # Persist extracted preferences
        if result.extracted_preferences:
            try:
                session.preferences = json.dumps(result.extracted_preferences)
            except (TypeError, ValueError):
                pass

        db.commit()
        db.refresh(assistant_msg)

        # Yield SSE events
        for event_type, data in events:
            yield _sse_event(event_type, data)

        yield _sse_event(
            "done",
            {
                "message_id": str(assistant_msg.id),
                "referenced_book_ids": result.referenced_book_ids,
                "model_used": result.model_used,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
