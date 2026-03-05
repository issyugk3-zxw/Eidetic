from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from eidetic.core.models import (
    CompactRequest,
    Document,
    ForgetRequest,
    MemoryEvent,
    OperationMeta,
    RecallQuery,
    RecallResult,
)


@runtime_checkable
class AsyncMemoryBackend(Protocol):
    system: str

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta: ...

    async def remember(self, event: MemoryEvent) -> OperationMeta: ...

    async def recall(self, query: RecallQuery) -> RecallResult: ...

    async def forget(self, request: ForgetRequest) -> OperationMeta: ...

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta: ...

    async def healthcheck(self) -> dict[str, Any]: ...


@runtime_checkable
class MemoryPlugin(Protocol):
    plugin_id: str
    version: str
    capabilities: set[str]
    required_dependencies: tuple[str, ...]
    install_hint: str

    async def create(self, config: dict[str, Any] | None = None) -> AsyncMemoryBackend: ...


class AsyncMemoryHandle:
    """Async wrapper exposed to users."""

    def __init__(self, backend: AsyncMemoryBackend, capabilities: set[str], system: str):
        self._backend = backend
        self._capabilities = capabilities
        self._system = system

    @property
    def system(self) -> str:
        return self._system

    @property
    def capabilities(self) -> set[str]:
        return set(self._capabilities)

    @property
    def backend(self) -> AsyncMemoryBackend:
        """Advanced escape hatch for plugin-specific APIs."""
        return self._backend

    def _require(self, capability: str) -> None:
        from eidetic.core.errors import CapabilityNotSupportedError

        if capability not in self._capabilities:
            raise CapabilityNotSupportedError(self._system, capability)

    async def _safe_call(self, operation: str, fn, *args, **kwargs):
        from eidetic.core.errors import BackendOperationError, EideticError

        try:
            return await fn(*args, **kwargs)
        except EideticError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise BackendOperationError(self._system, operation, str(exc)) from exc

    async def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        self._require("ingest")
        return await self._safe_call("ingest", self._backend.ingest, documents)

    async def remember(self, event: MemoryEvent) -> OperationMeta:
        self._require("remember")
        return await self._safe_call("remember", self._backend.remember, event)

    async def recall(self, query: RecallQuery) -> RecallResult:
        self._require("recall")
        return await self._safe_call("recall", self._backend.recall, query)

    async def forget(self, request: ForgetRequest) -> OperationMeta:
        self._require("forget")
        return await self._safe_call("forget", self._backend.forget, request)

    async def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        self._require("compact")
        return await self._safe_call("compact", self._backend.compact, request)

    async def healthcheck(self) -> dict[str, Any]:
        return await self._safe_call("healthcheck", self._backend.healthcheck)


class MemoryHandle:
    """Synchronous wrapper around AsyncMemoryHandle."""

    def __init__(self, async_handle: AsyncMemoryHandle):
        self._async_handle = async_handle

    @property
    def system(self) -> str:
        return self._async_handle.system

    @property
    def capabilities(self) -> set[str]:
        return self._async_handle.capabilities

    @property
    def async_handle(self) -> AsyncMemoryHandle:
        """Advanced access to async/common backend surface."""
        return self._async_handle

    def _run(self, coro):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "Cannot use synchronous MemoryHandle inside an active event loop. "
            "Use AsyncMemoryHandle instead."
        )

    def ingest(self, documents: list[Document] | Document) -> OperationMeta:
        return self._run(self._async_handle.ingest(documents))

    def remember(self, event: MemoryEvent) -> OperationMeta:
        return self._run(self._async_handle.remember(event))

    def recall(self, query: RecallQuery) -> RecallResult:
        return self._run(self._async_handle.recall(query))

    def forget(self, request: ForgetRequest) -> OperationMeta:
        return self._run(self._async_handle.forget(request))

    def compact(self, request: CompactRequest | None = None) -> OperationMeta:
        return self._run(self._async_handle.compact(request))

    def healthcheck(self) -> dict[str, Any]:
        return self._run(self._async_handle.healthcheck())
