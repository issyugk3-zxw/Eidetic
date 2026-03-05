from __future__ import annotations

"""
HippoRAG (https://github.com/OSU-NLP-Group/HippoRAG) native backend.

HippoRAG models long-term memory using a knowledge-graph approach inspired
by human hippocampal memory consolidation.  It requires an LLM for named
entity recognition and an embedding model for retrieval.

Installation
------------
HippoRAG is not on PyPI under a stable name.  Install from GitHub::

    pip install "git+https://github.com/OSU-NLP-Group/HippoRAG.git"

or if a PyPI release is available::

    pip install hipporag

Config keys (under ``config["plugin_config"]``):
    save_dir        str  Where HippoRAG persists its graph.
                         Default: ``./hipporag_data``
    llm_model       str  LLM model name.   Default: ``gpt-4o-mini``
    embedding_model str  Embedding model.  Default: ``text-embedding-3-small``
    api_key         str  OpenAI API key (reads OPENAI_API_KEY if absent).

Note on ``forget`` / ``compact``
---------------------------------
HippoRAG rebuilds its graph from the full corpus.  Forgetting a document
requires rebuilding the index from scratch.  We maintain a SQLite sidecar
for ID tracking; call ``compact()`` to trigger a full re-index that drops
forgotten documents.
"""

import json
import sqlite3
import time
from pathlib import Path
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


class HippoRAGBackend:
    """Eidetic backend wrapping HippoRAG."""

    def __init__(self, hipporag: Any, save_dir: Path) -> None:
        self.system = "hipporag"
        self._hr = hipporag
        self._save_dir = save_dir
        self._sidecar = save_dir / "eid_index.db"
        self._init_sidecar()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: dict[str, Any]) -> "HippoRAGBackend":
        import os

        try:
            from hipporag import HippoRAG
        except ImportError:
            raise ImportError(
                "HippoRAG is not installed. Install with:\n"
                '  pip install "git+https://github.com/OSU-NLP-Group/HippoRAG.git"\n'
                "or: pip install hipporag"
            )

        pc = config.get("plugin_config", {})
        save_dir = Path(pc.get("save_dir", "./hipporag_data"))
        save_dir.mkdir(parents=True, exist_ok=True)

        llm_model = pc.get("llm_model", "gpt-4o-mini")
        embed_model = pc.get("embedding_model", "text-embedding-3-small")
        api_key = pc.get("api_key")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

        # HippoRAG constructor varies between versions; try common shapes
        try:
            hr = HippoRAG(
                project_name=str(save_dir),
                global_config={
                    "llm_model": llm_model,
                    "embedding_model": embed_model,
                },
            )
        except TypeError:
            try:
                hr = HippoRAG(
                    save_dir=str(save_dir),
                    llm_model_name=llm_model,
                    embedding_model_name=embed_model,
                )
            except TypeError:
                hr = HippoRAG(save_dir=str(save_dir))

        return cls(hr, save_dir)

    # ------------------------------------------------------------------
    # Sidecar (ID + content tracking for rebuild support)
    # ------------------------------------------------------------------

    def _sidecar_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._sidecar))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sidecar(self) -> None:
        with self._sidecar_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eid_docs (
                    id      TEXT PRIMARY KEY,
                    kind    TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags    TEXT NOT NULL DEFAULT '[]',
                    deleted INTEGER NOT NULL DEFAULT 0
                )
            """)

    def _sidecar_upsert(
        self, id: str, kind: str, content: str, tags: list[str]
    ) -> None:
        with self._sidecar_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eid_docs(id, kind, content, tags, deleted) "
                "VALUES (?,?,?,?,0)",
                [id, kind, content, json.dumps(tags)],
            )

    def _live_contents(self) -> list[str]:
        with self._sidecar_conn() as conn:
            rows = conn.execute(
                "SELECT content FROM eid_docs WHERE deleted=0"
            ).fetchall()
        return [r["content"] for r in rows]

    # ------------------------------------------------------------------
    # AsyncMemoryBackend protocol
    # ------------------------------------------------------------------

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        start = time.perf_counter()
        docs = documents if isinstance(documents, list) else [documents]

        new_texts = [d.content for d in docs]
        for d in docs:
            self._sidecar_upsert(d.id, "document", d.content, list(d.tags))

        # HippoRAG indexes all docs together; pass full live corpus
        all_texts = self._live_contents()
        self._hr.get_ready(docs=all_texts)

        return OperationMeta(
            operation="ingest",
            processed=len(docs),
            affected_ids=[d.id for d in docs],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"total_indexed": len(all_texts)},
        )

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        start = time.perf_counter()
        content = f"[{event.role}] {event.content}"
        self._sidecar_upsert(event.id, "event", content, list(event.tags))
        all_texts = self._live_contents()
        self._hr.get_ready(docs=all_texts)
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

        try:
            results = self._hr.retrieve(queries=[query.query], num_to_retrieve=top_k)
        except TypeError:
            # Fallback for APIs that use different parameter names
            results = self._hr.retrieve(
                query=query.query, top_k=top_k
            )

        items: list[MemoryItem] = []
        # HippoRAG returns list[list[tuple[str, float]]] or similar
        raw = results[0] if results and isinstance(results[0], list) else results
        for i, entry in enumerate(raw[:top_k]):
            if isinstance(entry, tuple):
                passage, score = entry[0], float(entry[1]) if len(entry) > 1 else None
            elif isinstance(entry, str):
                passage, score = entry, None
            else:
                passage, score = str(entry), None

            items.append(
                MemoryItem(
                    id=str(uuid4()),
                    content=passage,
                    kind="document",
                    score=score,
                    provenance={"system": "hipporag"},
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

        with self._sidecar_conn() as conn:
            rows = conn.execute(
                "SELECT id, tags FROM eid_docs WHERE deleted=0"
            ).fetchall()

        to_delete: list[str] = []
        for row in rows:
            eid, row_tags = row["id"], json.loads(row["tags"])
            if request.ids and eid in request.ids:
                to_delete.append(eid)
            elif request.tags and set(request.tags).intersection(row_tags):
                to_delete.append(eid)

        if to_delete:
            ph = ",".join("?" * len(to_delete))
            with self._sidecar_conn() as conn:
                conn.execute(
                    f"UPDATE eid_docs SET deleted=1 WHERE id IN ({ph})", to_delete
                )
            removed = to_delete

        return OperationMeta(
            operation="forget",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={
                "note": (
                    "Records soft-deleted in sidecar. "
                    "Call compact() to rebuild HippoRAG index without them."
                )
            },
        )

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        """Rebuild HippoRAG index from surviving documents."""
        start = time.perf_counter()

        with self._sidecar_conn() as conn:
            dead = conn.execute(
                "SELECT id FROM eid_docs WHERE deleted=1"
            ).fetchall()

        dead_ids = [r["id"] for r in dead]

        if dead_ids:
            ph = ",".join("?" * len(dead_ids))
            with self._sidecar_conn() as conn:
                conn.execute(f"DELETE FROM eid_docs WHERE id IN ({ph})", dead_ids)

        all_texts = self._live_contents()
        if all_texts:
            self._hr.get_ready(docs=all_texts)

        return OperationMeta(
            operation="compact",
            processed=len(dead_ids),
            affected_ids=dead_ids,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"remaining": len(all_texts)},
        )

    async def healthcheck(self) -> dict[str, Any]:
        with self._sidecar_conn() as conn:
            live = conn.execute(
                "SELECT COUNT(*) FROM eid_docs WHERE deleted=0"
            ).fetchone()[0]
        return {
            "status": "ok",
            "system": self.system,
            "save_dir": str(self._save_dir),
            "records": live,
        }
