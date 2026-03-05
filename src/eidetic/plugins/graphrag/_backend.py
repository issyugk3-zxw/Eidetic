from __future__ import annotations

"""
Microsoft GraphRAG (https://github.com/microsoft/graphrag) native backend.

GraphRAG is a *batch* pipeline: it indexes documents offline and then
supports local / global graph-augmented search.  This backend wraps that
workflow as cleanly as possible inside the Eidetic interface.

Ingestion flow
--------------
Documents are written to ``<working_dir>/input/`` as plain-text files.
Indexing is NOT triggered automatically (it is expensive and long-running).
After ingesting your documents, call ``await backend.build_index()`` via
the escape hatch:

    handle = await manager.acreate("graphrag", config={...})
    await handle.backend.build_index()   # one-time setup

Recall works against whatever index currently exists on disk.  If no
index has been built yet, recall returns an empty result with a warning.

Config keys (under ``config["plugin_config"]``):
    working_dir     str   Root directory for GraphRAG data.
                          Default: ``./graphrag_data``
    query_mode      str   "local" or "global".  Default: ``local``
    community_level int   Community level for global search.  Default: 2
    response_type   str   Answer format.  Default: ``"Multiple Paragraphs"``
"""

import json
import shutil
import sqlite3
import subprocess
import sys
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


class GraphRAGBackend:
    """Eidetic backend wrapping Microsoft GraphRAG."""

    def __init__(
        self,
        working_dir: Path,
        query_mode: str,
        community_level: int,
        response_type: str,
    ) -> None:
        self.system = "graphrag"
        self._working_dir = working_dir
        self._input_dir = working_dir / "input"
        self._output_dir = working_dir / "output"
        self._query_mode = query_mode
        self._community_level = community_level
        self._response_type = response_type
        self._sidecar = working_dir / "eid_index.db"

        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._init_sidecar()
        self._ensure_settings()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: dict[str, Any]) -> "GraphRAGBackend":
        pc = config.get("plugin_config", {})
        working_dir = Path(pc.get("working_dir", "./graphrag_data"))
        return cls(
            working_dir=working_dir,
            query_mode=pc.get("query_mode", "local"),
            community_level=int(pc.get("community_level", 2)),
            response_type=pc.get("response_type", "Multiple Paragraphs"),
        )

    # ------------------------------------------------------------------
    # Sidecar (ID tracking + soft delete)
    # ------------------------------------------------------------------

    def _sidecar_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._sidecar))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sidecar(self) -> None:
        with self._sidecar_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eid_docs (
                    id       TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    kind     TEXT NOT NULL,
                    tags     TEXT NOT NULL DEFAULT '[]',
                    deleted  INTEGER NOT NULL DEFAULT 0
                )
            """)

    def _sidecar_upsert(
        self, id: str, filename: str, kind: str, tags: list[str]
    ) -> None:
        with self._sidecar_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eid_docs(id, filename, kind, tags, deleted) "
                "VALUES (?,?,?,?,0)",
                [id, filename, kind, json.dumps(tags)],
            )

    # ------------------------------------------------------------------
    # GraphRAG settings bootstrap
    # ------------------------------------------------------------------

    def _ensure_settings(self) -> None:
        """Write a minimal settings.yaml if one does not exist."""
        settings_file = self._working_dir / "settings.yaml"
        if settings_file.exists():
            return
        settings_file.write_text(
            "# GraphRAG settings - edit as needed\n"
            "# See https://microsoft.github.io/graphrag/config/\n\n"
            "input:\n"
            "  type: file\n"
            "  file_type: text\n"
            "  base_dir: input\n\n"
            "output:\n"
            "  type: file\n"
            "  base_dir: output\n\n"
            "llm:\n"
            "  type: openai_chat\n"
            "  model: gpt-4o-mini\n\n"
            "embeddings:\n"
            "  llm:\n"
            "    type: openai_embedding\n"
            "    model: text-embedding-3-small\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # AsyncMemoryBackend protocol
    # ------------------------------------------------------------------

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        start = time.perf_counter()
        docs = documents if isinstance(documents, list) else [documents]
        for doc in docs:
            filename = f"{doc.id}.txt"
            (self._input_dir / filename).write_text(doc.content, encoding="utf-8")
            self._sidecar_upsert(doc.id, filename, "document", list(doc.tags))
        return OperationMeta(
            operation="ingest",
            processed=len(docs),
            affected_ids=[d.id for d in docs],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={
                "note": (
                    "Documents written to input/. "
                    "Call backend.build_index() to make them searchable."
                ),
                "working_dir": str(self._working_dir),
            },
        )

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        start = time.perf_counter()
        filename = f"{event.id}.txt"
        content = f"[{event.role}] {event.content}"
        (self._input_dir / filename).write_text(content, encoding="utf-8")
        self._sidecar_upsert(event.id, filename, "event", list(event.tags))
        return OperationMeta(
            operation="remember",
            processed=1,
            affected_ids=[event.id],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def recall(self, query: RecallQuery) -> RecallResult:
        start = time.perf_counter()

        if not self._output_dir.exists() or not any(self._output_dir.iterdir()):
            return RecallResult(
                query=query.query,
                items=[],
                latency_ms=int((time.perf_counter() - start) * 1000),
                meta={
                    "warning": (
                        "No index found. Call backend.build_index() first, "
                        "or run: graphrag index --root <working_dir>"
                    )
                },
            )

        try:
            result_text = await self._run_query(query.query)
        except Exception as exc:
            result_text = f"[GraphRAG query error: {exc}]"

        items: list[MemoryItem] = []
        if result_text and result_text.strip():
            items.append(
                MemoryItem(
                    id=str(uuid4()),
                    content=result_text.strip(),
                    kind="document",
                    score=1.0,
                    provenance={
                        "system": "graphrag",
                        "mode": self._query_mode,
                        "working_dir": str(self._working_dir),
                    },
                )
            )

        return RecallResult(
            query=query.query,
            items=items,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def _run_query(self, query: str) -> str:
        """Run graphrag query via CLI subprocess (most stable across versions)."""
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "graphrag",
            "query",
            "--root",
            str(self._working_dir),
            "--method",
            self._query_mode,
            "--query",
            query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        return stdout.decode(errors="replace").strip()

    async def forget(self, request: ForgetRequest) -> OperationMeta:
        start = time.perf_counter()
        removed: list[str] = []

        with self._sidecar_conn() as conn:
            rows = conn.execute(
                "SELECT id, filename, tags FROM eid_docs WHERE deleted=0"
            ).fetchall()

        to_delete: list[tuple[str, str]] = []  # (id, filename)
        for row in rows:
            eid, fname, tags_json = row["id"], row["filename"], row["tags"]
            row_tags: list[str] = json.loads(tags_json)

            should = False
            if request.ids and eid in request.ids:
                should = True
            elif request.tags and set(request.tags).intersection(row_tags):
                should = True
            if should:
                to_delete.append((eid, fname))

        for eid, fname in to_delete:
            fpath = self._input_dir / fname
            if fpath.exists():
                if request.hard_delete:
                    fpath.unlink()
                # else: leave file; marked deleted in sidecar

        if to_delete:
            ids = [t[0] for t in to_delete]
            ph = ",".join("?" * len(ids))
            with self._sidecar_conn() as conn:
                if request.hard_delete:
                    conn.execute(f"DELETE FROM eid_docs WHERE id IN ({ph})", ids)
                else:
                    conn.execute(
                        f"UPDATE eid_docs SET deleted=1 WHERE id IN ({ph})", ids
                    )
            removed = ids

        return OperationMeta(
            operation="forget",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={
                "note": (
                    "Call backend.build_index() to apply deletions to the graph index."
                )
            },
        )

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        """Remove soft-deleted input files and rebuild the index."""
        start = time.perf_counter()
        removed: list[str] = []

        with self._sidecar_conn() as conn:
            dead = conn.execute(
                "SELECT id, filename FROM eid_docs WHERE deleted=1"
            ).fetchall()

        for row in dead:
            fpath = self._input_dir / row["filename"]
            if fpath.exists():
                fpath.unlink()
            removed.append(row["id"])

        if removed:
            with self._sidecar_conn() as conn:
                ph = ",".join("?" * len(removed))
                conn.execute(f"DELETE FROM eid_docs WHERE id IN ({ph})", removed)

        # Rebuild the index with remaining documents
        await self.build_index()

        return OperationMeta(
            operation="compact",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def build_index(self) -> dict[str, Any]:
        """
        Trigger the GraphRAG indexing pipeline.

        This is a long-running operation.  Call it once after initial data
        loading, and again whenever you want to incorporate new documents.

        Returns a dict with ``success``, ``output`` (last 1 000 chars of stdout),
        and ``error`` (stderr if any).
        """
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "graphrag",
            "index",
            "--root",
            str(self._working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "success": proc.returncode == 0,
            "output": stdout.decode(errors="replace")[-1000:],
            "error": stderr.decode(errors="replace")[-500:] if proc.returncode != 0 else "",
        }

    async def healthcheck(self) -> dict[str, Any]:
        with self._sidecar_conn() as conn:
            live = conn.execute(
                "SELECT COUNT(*) FROM eid_docs WHERE deleted=0"
            ).fetchone()[0]
        index_built = (
            self._output_dir.exists()
            and any(self._output_dir.iterdir())
        )
        return {
            "status": "ok",
            "system": self.system,
            "working_dir": str(self._working_dir),
            "query_mode": self._query_mode,
            "index_built": index_built,
            "records_staged": live,
        }
