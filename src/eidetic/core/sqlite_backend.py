from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _check_fts5() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _t USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


_FTS5_AVAILABLE = _check_fts5()


class SqliteBackend:
    """
    Persistent memory backend using Python's built-in sqlite3.
    Zero external dependencies.  Uses FTS5 for ranked full-text search
    when available, with automatic LIKE-based fallback.

    WAL journal mode is enabled so multiple readers don't block writers.
    Soft-delete is supported: records marked deleted=1 are invisible to
    recall/healthcheck but can be permanently removed with compact().
    """

    def __init__(self, system: str, db_path: str | Path = "eidetic.db"):
        self.system = system
        self._db_path = str(db_path)
        self._use_fts5 = _FTS5_AVAILABLE
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA wal_autocheckpoint=100")
        return conn

    def close(self) -> None:
        """Checkpoint and release WAL files.  Call when done with this backend."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id          TEXT PRIMARY KEY,
                    kind        TEXT NOT NULL CHECK(kind IN ('document','event')),
                    content     TEXT NOT NULL,
                    tags        TEXT NOT NULL DEFAULT '[]',
                    metadata    TEXT NOT NULL DEFAULT '{}',
                    created_at  TEXT NOT NULL,
                    deleted     INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kind       ON records(kind)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON records(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deleted    ON records(deleted)"
            )
            if self._use_fts5:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS records_fts
                    USING fts5(id UNINDEXED, content, tokenize='unicode61')
                """)

    def _upsert(
        self,
        conn: sqlite3.Connection,
        id: str,
        kind: str,
        content: str,
        tags: list[str],
        metadata: dict[str, Any],
        created_at: datetime,
    ) -> None:
        existing = conn.execute(
            "SELECT id FROM records WHERE id=?", [id]
        ).fetchone()
        conn.execute(
            "INSERT OR REPLACE INTO records "
            "(id, kind, content, tags, metadata, created_at, deleted) "
            "VALUES (?,?,?,?,?,?,0)",
            [
                id,
                kind,
                content,
                json.dumps(tags),
                json.dumps(metadata),
                created_at.isoformat(),
            ],
        )
        if self._use_fts5:
            if existing:
                conn.execute("DELETE FROM records_fts WHERE id=?", [id])
            conn.execute(
                "INSERT INTO records_fts(id, content) VALUES (?,?)", [id, content]
            )

    def _delete_ids(
        self, conn: sqlite3.Connection, ids: list[str], hard: bool
    ) -> list[str]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        if hard:
            existing = [
                r[0]
                for r in conn.execute(
                    f"SELECT id FROM records WHERE id IN ({ph})", ids
                ).fetchall()
            ]
            conn.execute(f"DELETE FROM records WHERE id IN ({ph})", ids)
            if self._use_fts5:
                conn.execute(f"DELETE FROM records_fts WHERE id IN ({ph})", ids)
        else:
            existing = [
                r[0]
                for r in conn.execute(
                    f"SELECT id FROM records WHERE id IN ({ph}) AND deleted=0", ids
                ).fetchall()
            ]
            conn.execute(
                f"UPDATE records SET deleted=1 WHERE id IN ({ph})", ids
            )
            # Keep FTS entries (filtered by JOIN in recall queries)
        return existing

    # ------------------------------------------------------------------
    # AsyncMemoryBackend protocol
    # ------------------------------------------------------------------

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        start = time.perf_counter()
        docs = documents if isinstance(documents, list) else [documents]
        with self._conn() as conn:
            for d in docs:
                self._upsert(
                    conn, d.id, "document", d.content,
                    list(d.tags), dict(d.metadata), d.created_at,
                )
        return OperationMeta(
            operation="ingest",
            processed=len(docs),
            affected_ids=[d.id for d in docs],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"backend": "sqlite", "fts5": self._use_fts5},
        )

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        start = time.perf_counter()
        meta = {**dict(event.metadata), "role": event.role}
        with self._conn() as conn:
            self._upsert(
                conn, event.id, "event", event.content,
                list(event.tags), meta, event.created_at,
            )
        return OperationMeta(
            operation="remember",
            processed=1,
            affected_ids=[event.id],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"backend": "sqlite", "fts5": self._use_fts5},
        )

    async def recall(self, query: RecallQuery) -> RecallResult:
        start = time.perf_counter()
        q = query.query.strip()

        # Build WHERE conditions for the base records table
        cond: list[str] = ["r.deleted = 0"]
        params: list[Any] = []

        if not query.include_documents:
            cond.append("r.kind != 'document'")
        if not query.include_events:
            cond.append("r.kind != 'event'")
        if query.since:
            cond.append("r.created_at >= ?")
            params.append(query.since.isoformat())
        if query.until:
            cond.append("r.created_at <= ?")
            params.append(query.until.isoformat())

        where = " AND ".join(cond)
        limit = max(query.top_k, 1)

        with self._conn() as conn:
            if q and self._use_fts5:
                # FTS5 MATCH with BM25 ranking (-bm25 gives positive score)
                sql = f"""
                    SELECT r.id, r.kind, r.content, r.tags, r.metadata,
                           r.created_at, -bm25(records_fts) AS score
                    FROM records_fts
                    JOIN records r ON r.id = records_fts.id
                    WHERE records_fts MATCH ? AND {where}
                    ORDER BY score DESC, r.created_at DESC
                    LIMIT ?
                """
                rows = conn.execute(sql, [q, *params, limit * 4]).fetchall()
            elif q:
                # LIKE fallback with word-overlap scoring
                sql = f"""
                    SELECT r.id, r.kind, r.content, r.tags, r.metadata,
                           r.created_at, 0.5 AS score
                    FROM records r
                    WHERE {where} AND r.content LIKE ?
                    ORDER BY r.created_at DESC
                    LIMIT ?
                """
                rows = conn.execute(
                    sql, [*params, f"%{q}%", limit * 4]
                ).fetchall()
            else:
                sql = f"""
                    SELECT r.id, r.kind, r.content, r.tags, r.metadata,
                           r.created_at, 1.0 AS score
                    FROM records r
                    WHERE {where}
                    ORDER BY r.created_at DESC
                    LIMIT ?
                """
                rows = conn.execute(sql, [*params, limit]).fetchall()

        # Tag filtering (JSON stored, done in Python)
        items: list[MemoryItem] = []
        for row in rows:
            row_tags: list[str] = json.loads(row["tags"])
            if query.tags and not set(query.tags).intersection(row_tags):
                continue
            items.append(
                MemoryItem(
                    id=row["id"],
                    content=row["content"],
                    kind=row["kind"],  # type: ignore[arg-type]
                    score=float(row["score"]),
                    tags=row_tags,
                    metadata=json.loads(row["metadata"]),
                    provenance={"system": self.system, "backend": "sqlite"},
                    created_at=_dt(row["created_at"]),
                )
            )

        return RecallResult(
            query=query.query,
            items=items[:limit],
            latency_ms=int((time.perf_counter() - start) * 1000),
            meta={"fts5": self._use_fts5, "total_candidates": len(items)},
        )

    async def forget(self, request: ForgetRequest) -> OperationMeta:
        start = time.perf_counter()
        removed: list[str] = []

        with self._conn() as conn:
            # 1. ID-based deletion
            if request.ids:
                removed.extend(
                    self._delete_ids(conn, list(request.ids), request.hard_delete)
                )

            # 2. Filter-based deletion (tags AND/OR date range)
            if request.tags or request.since or request.until:
                cond: list[str] = ["deleted=0"]
                params: list[Any] = []
                if request.since:
                    cond.append("created_at >= ?")
                    params.append(request.since.isoformat())
                if request.until:
                    cond.append("created_at <= ?")
                    params.append(request.until.isoformat())

                rows = conn.execute(
                    "SELECT id, tags FROM records WHERE " + " AND ".join(cond),
                    params,
                ).fetchall()

                candidates: list[str] = []
                for row in rows:
                    if row["id"] in removed:
                        continue
                    row_tags: list[str] = json.loads(row["tags"])
                    # tags filter: delete if ANY requested tag matches
                    if request.tags and not set(request.tags).intersection(row_tags):
                        continue
                    candidates.append(row["id"])

                if candidates:
                    extra = self._delete_ids(conn, candidates, request.hard_delete)
                    removed.extend(extra)

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
        removed: list[str] = []

        with self._conn() as conn:
            # Trim to max_items (keep N most recent live records)
            if request.max_items is not None and request.max_items >= 0:
                overflow = conn.execute(
                    "SELECT id FROM records WHERE deleted=0 "
                    "ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                    [request.max_items],
                ).fetchall()
                overflow_ids = [r[0] for r in overflow]
                if overflow_ids:
                    removed.extend(
                        self._delete_ids(conn, overflow_ids, hard=True)
                    )

            # Purge all soft-deleted records
            soft_deleted = conn.execute(
                "SELECT id FROM records WHERE deleted=1"
            ).fetchall()
            sd_ids = [r[0] for r in soft_deleted]
            if sd_ids:
                ph = ",".join("?" * len(sd_ids))
                conn.execute(f"DELETE FROM records WHERE id IN ({ph})", sd_ids)
                if self._use_fts5:
                    conn.execute(
                        f"DELETE FROM records_fts WHERE id IN ({ph})", sd_ids
                    )

            remaining = conn.execute(
                "SELECT COUNT(*) FROM records WHERE deleted=0"
            ).fetchone()[0]

            # Reclaim disk space
            conn.execute("VACUUM")

        return OperationMeta(
            operation="compact",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"strategy": request.strategy, "remaining": remaining},
        )

    async def healthcheck(self) -> dict[str, Any]:
        with self._conn() as conn:
            live = conn.execute(
                "SELECT COUNT(*) FROM records WHERE deleted=0"
            ).fetchone()[0]
            soft = conn.execute(
                "SELECT COUNT(*) FROM records WHERE deleted=1"
            ).fetchone()[0]
        return {
            "status": "ok",
            "system": self.system,
            "backend": "sqlite",
            "db_path": self._db_path,
            "fts5": self._use_fts5,
            "records": live,
            "soft_deleted": soft,
        }
