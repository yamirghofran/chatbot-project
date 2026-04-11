"""Chat session and message endpoints with SSE streaming."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from bookdb.db.chat_models import ChatMessage, ChatSession
from bookdb.db.models import User
from bookdb.models.chatbot_llm import create_groq_client_sync

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


def _serialize_message(msg: ChatMessage) -> dict[str, Any]:
    tool_trace = None
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
        tool_trace = {
            "tool": msg.tool_name,
            "input": tool_input,
            "output": tool_output,
            "source": None,
        }
        if isinstance(tool_output, dict):
            tool_trace["source"] = tool_output.get("source")

    ref_ids: list[int] = []
    if msg.referenced_book_ids:
        try:
            ref_ids = json.loads(msg.referenced_book_ids)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "id": str(msg.id),
        "role": msg.role,
        "content": msg.content,
        "toolName": msg.tool_name,
        "toolTrace": tool_trace,
        "referencedBookIds": ref_ids,
        "modelUsed": msg.model_used,
        "timestamp": msg.created_at.isoformat() if msg.created_at else "",
    }


def _get_session_or_404(
    db: Session, session_id: int, user: User | None,
) -> ChatSession:
    session = db.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.is_active.is_(True),
        )
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id is not None and user is not None and session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your session")
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

    prefs = None
    if session.preferences:
        try:
            prefs = json.loads(session.preferences)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "id": str(session.id),
        "title": session.title,
        "messages": [_serialize_message(m) for m in messages],
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

        # Persist assistant message
        tool_name = None
        tool_input_json = None
        tool_output_json = None
        if result.tool_calls:
            first_tc = result.tool_calls[0]
            tool_name = first_tc.name
            try:
                tool_input_json = json.dumps(first_tc.input)
            except (TypeError, ValueError):
                pass
            try:
                tool_output_json = json.dumps(first_tc.output, default=str)
            except (TypeError, ValueError):
                pass

        ref_ids_json = json.dumps(result.referenced_book_ids) if result.referenced_book_ids else None

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

        db.commit()
        db.refresh(assistant_msg)

        # Yield SSE events
        for event_type, data in events:
            yield _sse_event(event_type, data)

        yield _sse_event("done", {
            "message_id": str(assistant_msg.id),
            "referenced_book_ids": result.referenced_book_ids,
            "model_used": result.model_used,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
