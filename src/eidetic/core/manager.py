from __future__ import annotations

from typing import Any

from eidetic.core.models import SystemInfo
from eidetic.core.protocols import AsyncMemoryHandle, MemoryHandle
from eidetic.core.registry import PluginRegistry


class MemoryManager:
    def __init__(self, registry: PluginRegistry | None = None):
        self._registry = registry or PluginRegistry()

    def list_systems(self) -> list[SystemInfo]:
        return self._registry.list_system_info()

    def get_system_info(self, system: str) -> SystemInfo:
        return self._registry.get_system_info(system)

    async def acreate(
        self, system: str, config: dict[str, Any] | None = None
    ) -> AsyncMemoryHandle:
        plugin = self._registry.get_plugin(system)
        backend = await plugin.create(config or {})
        return AsyncMemoryHandle(
            backend=backend,
            capabilities=set(plugin.capabilities),
            system=plugin.plugin_id,
        )

    def create(self, system: str, config: dict[str, Any] | None = None) -> MemoryHandle:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            async_handle = asyncio.run(self.acreate(system, config=config or {}))
            return MemoryHandle(async_handle)
        raise RuntimeError(
            "Cannot call MemoryManager.create inside an active event loop. "
            "Use MemoryManager.acreate instead."
        )
