from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(BaseModel):
    id: str = Field(default_factory=_new_id)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class MemoryEvent(BaseModel):
    id: str = Field(default_factory=_new_id)
    content: str
    role: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class RecallQuery(BaseModel):
    query: str
    top_k: int = 5
    tags: list[str] = Field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    include_documents: bool = True
    include_events: bool = True


class MemoryItem(BaseModel):
    id: str
    content: str
    kind: Literal["document", "event"]
    score: float | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class RecallResult(BaseModel):
    query: str
    items: list[MemoryItem] = Field(default_factory=list)
    latency_ms: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class ForgetRequest(BaseModel):
    ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None
    hard_delete: bool = False


class CompactRequest(BaseModel):
    strategy: str = "default"
    max_items: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationMeta(BaseModel):
    operation: str
    success: bool = True
    processed: int = 0
    affected_ids: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    backend: str
    details: dict[str, Any] = Field(default_factory=dict)


class SystemInfo(BaseModel):
    system: str
    plugin_id: str
    version: str
    capabilities: list[str] = Field(default_factory=list)
    required_dependencies: list[str] = Field(default_factory=list)
    installed: bool = False
    install_hint: str = ""
