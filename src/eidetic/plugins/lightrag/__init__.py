from __future__ import annotations

from typing import Any

from eidetic.core.plugin_base import BasePlugin
from eidetic.core.protocols import AsyncMemoryBackend

_CAPABILITIES = {"ingest", "remember", "recall", "forget", "compact"}


class LightRAGPlugin(BasePlugin):
    plugin_id = "lightrag"
    version = "0.1.0"
    capabilities = _CAPABILITIES
    # lightrag-hku installs as the "lightrag" module
    required_dependencies = ("lightrag",)
    install_hint = 'pip install "eidetic[lightrag]"'

    async def _build_native_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        from eidetic.plugins.lightrag._backend import LightRAGBackend

        return await LightRAGBackend.create(config)


def get_plugin() -> LightRAGPlugin:
    return LightRAGPlugin()
