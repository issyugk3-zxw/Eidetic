from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from eidetic.core.models import (
    CompactRequest,
    Document,
    ForgetRequest,
    MemoryEvent,
    MemoryItem,
    OperationMeta,
    RecallQuery,
    RecallResult,
)


@dataclass
class _StoredRecord:
    kind: Literal["document", "event"]
    id: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime


class InMemorySemanticBackend:
    def __init__(self, system: str, native: bool = False):
        self.system = system
        self._native = native
        self._records: dict[str, _StoredRecord] = {}

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        start = time.perf_counter()
        docs = documents if isinstance(documents, list) else [documents]
        for doc in docs:
            self._records[doc.id] = _StoredRecord(
                kind="document",
                id=doc.id,
                content=doc.content,
                tags=list(doc.tags),
                metadata=dict(doc.metadata),
                created_at=doc.created_at,
            )
        return OperationMeta(
            operation="ingest",
            processed=len(docs),
            affected_ids=[d.id for d in docs],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"mode": "native" if self._native else "mock"},
        )

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        start = time.perf_counter()
        self._records[event.id] = _StoredRecord(
            kind="event",
            id=event.id,
            content=event.content,
            tags=list(event.tags),
            metadata=dict(event.metadata),
            created_at=event.created_at,
        )
        return OperationMeta(
            operation="remember",
            processed=1,
            affected_ids=[event.id],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"mode": "native" if self._native else "mock"},
        )

    async def recall(self, query: RecallQuery) -> RecallResult:
        start = time.perf_counter()
        q = query.query.strip().lower()
        items: list[MemoryItem] = []
        for record in self._records.values():
            if record.kind == "document" and not query.include_documents:
                continue
            if record.kind == "event" and not query.include_events:
                continue
            if query.tags and not set(query.tags).intersection(record.tags):
                continue
            if query.since and record.created_at < query.since:
                continue
            if query.until and record.created_at > query.until:
                continue

            if q:
                score = self._score(q, record.content.lower())
                if score <= 0:
                    continue
            else:
                # Empty query means "recent memory recall".
                score = 1.0
            items.append(
                MemoryItem(
                    id=record.id,
                    content=record.content,
                    kind=record.kind,
                    score=score,
                    tags=list(record.tags),
                    metadata=dict(record.metadata),
                    provenance={
                        "system": self.system,
                        "mode": "native" if self._native else "mock",
                    },
                    created_at=record.created_at,
                )
            )
        if q:
            items.sort(
                key=lambda x: ((x.score or 0), x.created_at.timestamp()),
                reverse=True,
            )
        else:
            items.sort(key=lambda x: x.created_at.timestamp(), reverse=True)
        top_items = items[: max(query.top_k, 1)]
        latency = int((time.perf_counter() - start) * 1000)
        return RecallResult(
            query=query.query,
            items=top_items,
            latency_ms=latency,
            meta={"total_candidates": len(items)},
        )

    async def forget(self, request: ForgetRequest) -> OperationMeta:
        start = time.perf_counter()
        removed: list[str] = []

        for item_id in request.ids:
            if item_id in self._records:
                del self._records[item_id]
                removed.append(item_id)

        if request.tags or request.since or request.until:
            extra_to_remove: list[str] = []
            for record in self._records.values():
                if request.tags and not set(request.tags).intersection(record.tags):
                    continue
                if request.since and record.created_at < request.since:
                    continue
                if request.until and record.created_at > request.until:
                    continue
                extra_to_remove.append(record.id)
            for item_id in extra_to_remove:
                self._records.pop(item_id, None)
                if item_id not in removed:
                    removed.append(item_id)

        return OperationMeta(
            operation="forget",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"hard_delete": request.hard_delete},
        )

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        start = time.perf_counter()
        request = request or CompactRequest()
        affected: list[str] = []

        if request.max_items is not None and request.max_items >= 0:
            ordered = sorted(
                self._records.values(),
                key=lambda r: r.created_at,
                reverse=True,
            )
            keep_ids = {record.id for record in ordered[: request.max_items]}
            for record in list(self._records.values()):
                if record.id not in keep_ids:
                    affected.append(record.id)
                    self._records.pop(record.id, None)

        return OperationMeta(
            operation="compact",
            processed=len(affected),
            affected_ids=affected,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"strategy": request.strategy, "remaining": len(self._records)},
        )

    async def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "system": self.system,
            "mode": "native" if self._native else "mock",
            "records": len(self._records),
        }

    @staticmethod
    def _score(query: str, content: str) -> float:
        if not query:
            return 0.0
        if query in content:
            return 1.0
        query_terms = [term for term in query.split() if term]
        if not query_terms:
            return 0.0
        content_terms = set(content.split())
        overlap = sum(1 for term in query_terms if term in content_terms)
        return overlap / max(len(query_terms), 1)
