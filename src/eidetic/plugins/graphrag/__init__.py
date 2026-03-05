from __future__ import annotations

from typing import Any

from eidetic.core.plugin_base import BasePlugin
from eidetic.core.protocols import AsyncMemoryBackend

_CAPABILITIES = {"ingest", "remember", "recall", "forget", "compact"}


class GraphRAGPlugin(BasePlugin):
    plugin_id = "graphrag"
    version = "0.1.0"
    capabilities = _CAPABILITIES
    required_dependencies = ("graphrag",)
    install_hint = 'pip install "eidetic[graphrag]"'

    async def _build_native_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        from eidetic.plugins.graphrag._backend import GraphRAGBackend

        return await GraphRAGBackend.create(config)


def get_plugin() -> GraphRAGPlugin:
    return GraphRAGPlugin()
