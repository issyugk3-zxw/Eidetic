from __future__ import annotations

"""
LightRAG (https://github.com/HKUDS/LightRAG) native backend.

LightRAG builds a knowledge graph + vector store and stores everything
on disk inside ``working_dir``.  It supports multiple retrieval modes:
``naive``, ``local``, ``global``, ``hybrid``, and ``mix``.

Because LightRAG uses an LLM for graph extraction and an embedding model
for vector search, you must supply the relevant credentials.

Config keys (under ``config["plugin_config"]``):
    working_dir     str   Where LightRAG persists its graph/vectors.
                          Default: ``./lightrag_data``
    mode            str   Retrieval mode passed to QueryParam.
                          One of: naive | local | global | hybrid | mix
                          Default: ``hybrid``
    llm_provider    str   "openai" | "anthropic" | "ollama" | "custom"
                          Default: ``openai``
    llm_model       str   Model name.  Default depends on provider.
    embedding_model str   Embedding model name (OpenAI only currently).
                          Default: ``text-embedding-3-small``
    embedding_dim   int   Dimension of embedding vectors.  Default: 1536
    api_key         str   API key (reads from env var if not supplied).

For ``llm_provider="custom"`` you may also pass:
    llm_func        callable  Async function ``async (prompt, **kw) -> str``
    embed_func      callable  Async function ``async (texts) -> list[list[float]]``

Note on ``forget``:
    LightRAG does not expose a public delete-by-document API.  We maintain
    a tiny SQLite sidecar (``<working_dir>/eid_index.db``) that maps Eidetic
    IDs to insertion status.  Forget marks records as deleted in the sidecar;
    they will NOT be removed from the LightRAG graph until a full re-index.
    Call ``backend.rebuild_index()`` via the escape hatch to achieve that.
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


class LightRAGBackend:
    """Eidetic backend wrapping LightRAG with filesystem persistence."""

    def __init__(self, rag: Any, working_dir: Path, mode: str) -> None:
        self.system = "lightrag"
        self._rag = rag
        self._working_dir = working_dir
        self._mode = mode
        self._sidecar = working_dir / "eid_index.db"
        self._init_sidecar()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: dict[str, Any]) -> "LightRAGBackend":
        from lightrag import LightRAG

        pc = config.get("plugin_config", {})
        working_dir = Path(pc.get("working_dir", "./lightrag_data"))
        working_dir.mkdir(parents=True, exist_ok=True)
        mode = pc.get("mode", "hybrid")

        llm_func, embed_func, embed_dim = cls._build_llm_and_embed(pc)

        try:
            from lightrag.utils import EmbeddingFunc

            rag = LightRAG(
                working_dir=str(working_dir),
                llm_model_func=llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embed_dim,
                    max_token_size=8192,
                    func=embed_func,
                ),
            )
        except TypeError:
            # Older LightRAG versions may have a different constructor
            rag = LightRAG(
                working_dir=str(working_dir),
                llm_model_func=llm_func,
            )

        return cls(rag, working_dir, mode)

    @staticmethod
    def _build_llm_and_embed(
        pc: dict[str, Any],
    ) -> tuple[Any, Any, int]:
        provider = pc.get("llm_provider", "openai")
        embed_dim = int(pc.get("embedding_dim", 1536))
        embed_model = pc.get("embedding_model", "text-embedding-3-small")
        api_key: str | None = pc.get("api_key")

        if provider == "openai":
            from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

            if api_key:
                import os
                os.environ.setdefault("OPENAI_API_KEY", api_key)

            async def embed_fn(texts: list[str]) -> list[list[float]]:
                return await openai_embed(
                    texts,
                    model=embed_model,
                )

            return gpt_4o_mini_complete, embed_fn, embed_dim

        if provider == "ollama":
            from lightrag.llm.ollama import ollama_model_complete, ollama_embed

            llm_model = pc.get("llm_model", "mistral")
            base_url = pc.get("base_url", "http://localhost:11434")

            async def llm_fn(prompt: str, **kw: Any) -> str:
                return await ollama_model_complete(
                    prompt, model=llm_model, host=base_url, **kw
                )

            async def embed_fn_ollama(texts: list[str]) -> list[list[float]]:
                return await ollama_embed(
                    texts, embed_model=embed_model, host=base_url
                )

            return llm_fn, embed_fn_ollama, embed_dim

        if provider == "anthropic":
            # LightRAG doesn't ship an Anthropic helper; build one manually
            import anthropic as _anthropic

            llm_model = pc.get("llm_model", "claude-haiku-4-5-20251001")
            _client = _anthropic.AsyncAnthropic(api_key=api_key)

            async def llm_fn_anthropic(prompt: str, **kw: Any) -> str:
                msg = await _client.messages.create(
                    model=llm_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text

            # Anthropic doesn't expose embeddings; fall back to a tiny local model
            try:
                from lightrag.llm.openai import openai_embed

                async def embed_fn_anthropic(texts: list[str]) -> list[list[float]]:
                    return await openai_embed(texts, model=embed_model)

                return llm_fn_anthropic, embed_fn_anthropic, embed_dim
            except ImportError:
                raise ValueError(
                    "When using provider='anthropic' you still need an OpenAI "
                    "key (or a local embedding model) for LightRAG's vector store. "
                    "Set llm_provider='anthropic' and also supply OPENAI_API_KEY, "
                    "or switch to provider='ollama' for a fully local setup."
                )

        if provider == "custom":
            llm_func = pc.get("llm_func")
            embed_func = pc.get("embed_func")
            if llm_func is None or embed_func is None:
                raise ValueError(
                    "provider='custom' requires 'llm_func' and 'embed_func' "
                    "in plugin_config."
                )
            return llm_func, embed_func, embed_dim

        raise ValueError(
            f"Unsupported llm_provider '{provider}'. "
            "Choose from: openai, anthropic, ollama, custom."
        )

    # ------------------------------------------------------------------
    # Sidecar index (lightweight ID tracking for delete support)
    # ------------------------------------------------------------------

    def _sidecar_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._sidecar))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sidecar(self) -> None:
        with self._sidecar_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eid_docs (
                    id         TEXT PRIMARY KEY,
                    kind       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    tags       TEXT NOT NULL DEFAULT '[]',
                    deleted    INTEGER NOT NULL DEFAULT 0
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

    def _sidecar_mark_deleted(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        with self._sidecar_conn() as conn:
            ph = ",".join("?" * len(ids))
            existing = [
                r[0]
                for r in conn.execute(
                    f"SELECT id FROM eid_docs WHERE id IN ({ph}) AND deleted=0", ids
                ).fetchall()
            ]
            if existing:
                conn.execute(
                    f"UPDATE eid_docs SET deleted=1 WHERE id IN ({ph})", existing
                )
        return existing

    # ------------------------------------------------------------------
    # AsyncMemoryBackend protocol
    # ------------------------------------------------------------------

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        start = time.perf_counter()
        docs = documents if isinstance(documents, list) else [documents]
        texts = [d.content for d in docs]
        await self._rag.ainsert(texts)
        for d in docs:
            self._sidecar_upsert(d.id, "document", d.content, list(d.tags))
        return OperationMeta(
            operation="ingest",
            processed=len(docs),
            affected_ids=[d.id for d in docs],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={"working_dir": str(self._working_dir)},
        )

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        start = time.perf_counter()
        await self._rag.ainsert(event.content)
        self._sidecar_upsert(event.id, "event", event.content, list(event.tags))
        return OperationMeta(
            operation="remember",
            processed=1,
            affected_ids=[event.id],
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def recall(self, query: RecallQuery) -> RecallResult:
        start = time.perf_counter()
        from lightrag import QueryParam

        result_text: str = await self._rag.aquery(
            query.query,
            param=QueryParam(mode=self._mode),
        )

        # LightRAG synthesises a single answer string; wrap as one MemoryItem
        items: list[MemoryItem] = []
        if result_text and result_text.strip():
            items.append(
                MemoryItem(
                    id=str(uuid4()),
                    content=result_text.strip(),
                    kind="document",
                    score=1.0,
                    provenance={
                        "system": "lightrag",
                        "mode": self._mode,
                        "working_dir": str(self._working_dir),
                    },
                )
            )

        return RecallResult(
            query=query.query,
            items=items,
            latency_ms=int((time.perf_counter() - start) * 1000),
            meta={
                "note": (
                    "LightRAG returns a synthesised answer, not raw passages. "
                    "One MemoryItem is returned containing the full response."
                )
            },
        )

    async def forget(self, request: ForgetRequest) -> OperationMeta:
        """
        Marks records as deleted in the sidecar index.

        LightRAG does not expose a public document-deletion API; records
        remain in the graph until ``rebuild_index()`` is called.
        """
        start = time.perf_counter()
        removed: list[str] = []

        if request.ids:
            removed.extend(self._sidecar_mark_deleted(list(request.ids)))

        if request.tags:
            with self._sidecar_conn() as conn:
                rows = conn.execute(
                    "SELECT id, tags FROM eid_docs WHERE deleted=0"
                ).fetchall()
            candidates = [
                r["id"]
                for r in rows
                if set(request.tags).intersection(json.loads(r["tags"]))
                and r["id"] not in removed
            ]
            removed.extend(self._sidecar_mark_deleted(candidates))

        return OperationMeta(
            operation="forget",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
            details={
                "note": (
                    "Records are soft-deleted in the sidecar index. "
                    "Call backend.rebuild_index() to purge from LightRAG graph."
                )
            },
        )

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        """Purge soft-deleted records and rebuild the LightRAG graph."""
        start = time.perf_counter()
        removed = await self.rebuild_index()
        return OperationMeta(
            operation="compact",
            processed=len(removed),
            affected_ids=removed,
            latency_ms=int((time.perf_counter() - start) * 1000),
            backend=self.system,
        )

    async def rebuild_index(self) -> list[str]:
        """
        Re-insert all non-deleted documents into a fresh LightRAG instance.
        Returns list of Eidetic IDs that were dropped (had been forgotten).
        """
        import shutil

        with self._sidecar_conn() as conn:
            live = conn.execute(
                "SELECT id, content FROM eid_docs WHERE deleted=0"
            ).fetchall()
            dead = conn.execute(
                "SELECT id FROM eid_docs WHERE deleted=1"
            ).fetchall()

        dead_ids = [r["id"] for r in dead]

        if dead_ids:
            # Wipe LightRAG storage and re-insert only live documents
            for sub in ["graph_chunk_entity_relation.graphml", "kv_store*",
                        "vdb_*", "vector*"]:
                for p in self._working_dir.glob(sub):
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)

            texts = [r["content"] for r in live]
            if texts:
                await self._rag.ainsert(texts)

            with self._sidecar_conn() as conn:
                conn.execute("DELETE FROM eid_docs WHERE deleted=1")

        return dead_ids

    async def healthcheck(self) -> dict[str, Any]:
        with self._sidecar_conn() as conn:
            live = conn.execute(
                "SELECT COUNT(*) FROM eid_docs WHERE deleted=0"
            ).fetchone()[0]
        return {
            "status": "ok",
            "system": self.system,
            "working_dir": str(self._working_dir),
            "mode": self._mode,
            "records": live,
        }
