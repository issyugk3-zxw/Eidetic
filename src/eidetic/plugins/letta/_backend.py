from __future__ import annotations

"""
Letta (https://github.com/letta-ai/letta) native backend.

Letta manages its own SQLite / PostgreSQL database and exposes a REST API.
You must have a Letta server running (default: http://localhost:8283).
Start one with: ``letta server``

Config keys (all optional, go under ``config["plugin_config"]``):
    base_url    str   URL of the Letta server.     Default: http://localhost:8283
    token       str   Bearer token if server auth is enabled.
    agent_name  str   Name of the Letta agent to use/create. Default: "eidetic"
"""

import time
from typing import Any
from uuid import uuid4

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

# Prefix embedded in every archived text so we can round-trip the Eidetic ID.
_EID_PREFIX = "[eid:"
_EID_SUFFIX = "] "


def _pack(eid: str, content: str, tags: list[str]) -> str:
    tag_str = f" [tags:{','.join(tags)}]" if tags else ""
    return f"{_EID_PREFIX}{eid}{_EID_SUFFIX}{content}{tag_str}"


def _unpack_id(text: str) -> str | None:
    if text.startswith(_EID_PREFIX):
        end = text.find(_EID_SUFFIX)
        if end != -1:
            return text[len(_EID_PREFIX) : end]
    return None


class LettaBackend:
    """Eidetic backend that delegates to a running Letta server."""

    def __init__(self, client: Any, agent_id: str) -> None:
        self.system = "letta"
        self._client = client
        self._agent_id = agent_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: dict[str, Any]) -> "LettaBackend":
        from letta import create_client

        pc = config.get("plugin_config", {})
        base_url: str = pc.get("base_url", "http://localhost:8283")
        token: str | None = pc.get("token")
        agent_name: str = pc.get("agent_name", "eidetic")

        client = create_client(base_url=base_url, token=token)

        # Reuse existing agent or create a fresh one
        agents = client.list_agents()
        existing = next((a for a in agents if a.name == agent_name), None)
        if existing:
            agent_id = existing.id
        else:
            try:
                from letta.schemas.memory import ChatMemory

                agent = client.create_agent(
                    name=agent_name,
                    memory=ChatMemory(
                        human="",
                        persona=(
                            "You are a persistent memory assistant. "
                            "You store and retrieve information faithfully."
                        ),
                    ),
                )
            except ImportError:
                # Fallback for alternative Letta API shapes
                agent = client.create_agent(name=agent_name)
            agent_id = agent.id

        return cls(client, agent_id)

    # ------------------------------------------------------------------
    # AsyncMemoryBackend protocol
    # ------------------------------------------------------------------

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        start = time.perf_counter()
        docs = documents if isinstance(documents, list) else [documents]
        for doc in docs:
            text = _pack(doc.id, doc.content, list(doc.tags))
            self._client.insert_archival_memory(
                agent_id=self._agent_id, memory=text
            )
        return OperationMeta(
            operation="ingest",
            processed=len(docs),
            affected_ids=[d.id for d in docs],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        start = time.perf_counter()
        text = _pack(event.id, f"[{event.role}] {event.content}", list(event.tags))
        self._client.insert_archival_memory(
            agent_id=self._agent_id, memory=text
        )
        return OperationMeta(
            operation="remember",
            processed=1,
            affected_ids=[event.id],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def recall(self, query: RecallQuery) -> RecallResult:
        start = time.perf_counter()
        top_k = max(query.top_k, 1)
        items: list[MemoryItem] = []

        try:
            # Letta REST API supports ?query= for semantic search
            results = self._client.get_archival_memory(
                agent_id=self._agent_id,
                query=query.query,
                limit=top_k,
            )
        except TypeError:
            # Older SDK versions don't accept query/limit
            results = self._client.get_archival_memory(
                agent_id=self._agent_id,
            )

        for i, mem in enumerate(results[:top_k]):
            raw_text: str = getattr(mem, "text", getattr(mem, "content", str(mem)))
            eid = _unpack_id(raw_text)
            # Strip our prefix for the returned content
            content = raw_text
            if eid:
                content = raw_text[len(_EID_PREFIX) + len(eid) + len(_EID_SUFFIX):]

            items.append(
                MemoryItem(
                    id=eid or str(getattr(mem, "id", uuid4())),
                    content=content,
                    kind="event",
                    score=None,
                    provenance={"system": "letta", "letta_id": str(getattr(mem, "id", ""))},
                )
            )

        return RecallResult(
            query=query.query,
            items=items,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def forget(self, request: ForgetRequest) -> OperationMeta:
        start = time.perf_counter()
        removed: list[str] = []

        if not request.ids and not request.tags and not request.since and not request.until:
            return OperationMeta(
                operation="forget",
                processed=0,
                affected_ids=[],
                latency_ms=0,
                backend=self.system,
            )

        # List all memories and filter client-side
        all_mems = self._client.get_archival_memory(agent_id=self._agent_id)
        for mem in all_mems:
            raw_text: str = getattr(mem, "text", getattr(mem, "content", str(mem)))
            eid = _unpack_id(raw_text)
            letta_id = str(getattr(mem, "id", ""))

            should_delete = False
            if eid and request.ids and eid in request.ids:
                should_delete = True
            elif request.tags:
                # Tags are embedded as "[tags:t1,t2]" at the end
                for tag in request.tags:
                    if f"[tags:" in raw_text and tag in raw_text:
                        should_delete = True
                        break

            if should_delete and letta_id:
                try:
                    self._client.delete_archival_memory(
                        agent_id=self._agent_id, memory_id=letta_id
                    )
                    if eid:
                        removed.append(eid)
                except Exception:
                    pass

        return OperationMeta(
            operation="forget",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"note": "Letta does not support hard vs soft delete."},
        )

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        start = time.perf_counter()
        request = request or CompactRequest()
        removed: list[str] = []

        if request.max_items is not None and request.max_items >= 0:
            all_mems = self._client.get_archival_memory(agent_id=self._agent_id)
            # Letta returns in insertion order; keep newest N
            to_drop = list(all_mems)[: max(0, len(all_mems) - request.max_items)]
            for mem in to_drop:
                letta_id = str(getattr(mem, "id", ""))
                raw_text = getattr(mem, "text", getattr(mem, "content", ""))
                eid = _unpack_id(raw_text) or letta_id
                try:
                    self._client.delete_archival_memory(
                        agent_id=self._agent_id, memory_id=letta_id
                    )
                    removed.append(eid)
                except Exception:
                    pass

        return OperationMeta(
            operation="compact",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def healthcheck(self) -> dict[str, Any]:
        try:
            agent = self._client.get_agent(self._agent_id)
            status = "ok"
            agent_name = getattr(agent, "name", "unknown")
        except Exception as exc:
            status = f"error: {exc}"
            agent_name = "unknown"
        return {
            "status": status,
            "system": self.system,
            "agent_id": self._agent_id,
            "agent_name": agent_name,
        }
