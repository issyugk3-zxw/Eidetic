from __future__ import annotations

from typing import Any

from eidetic.core.plugin_base import BasePlugin
from eidetic.core.protocols import AsyncMemoryBackend

_CAPABILITIES = {"ingest", "remember", "recall", "forget", "compact"}


class LettaPlugin(BasePlugin):
    plugin_id = "letta"
    version = "0.1.0"
    capabilities = _CAPABILITIES
    required_dependencies = ("letta",)
    install_hint = 'pip install "eidetic[letta]"'

    async def _build_native_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        from eidetic.plugins.letta._backend import LettaBackend

        return await LettaBackend.create(config)


def get_plugin() -> LettaPlugin:
    return LettaPlugin()
