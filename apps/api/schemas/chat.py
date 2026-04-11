from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SessionOut(BaseModel):
    id: str
    title: str
    createdAt: str
    updatedAt: str


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class ToolTraceOut(BaseModel):
    tool: str
    input: dict | None = None
    output: dict | None = None
    source: str | None = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    toolName: str | None = None
    toolTrace: ToolTraceOut | None = None
    referencedBookIds: list[int] = []
    modelUsed: str | None = None
    timestamp: str


class SessionDetailOut(BaseModel):
    id: str
    title: str
    messages: list[MessageOut] = []
    preferences: dict | None = None
    createdAt: str
