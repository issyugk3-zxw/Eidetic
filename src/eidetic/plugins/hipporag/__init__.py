from __future__ import annotations

from typing import Any

from eidetic.core.plugin_base import BasePlugin
from eidetic.core.protocols import AsyncMemoryBackend

_CAPABILITIES = {"ingest", "remember", "recall", "forget", "compact"}


class HippoRAGPlugin(BasePlugin):
    plugin_id = "hipporag"
    version = "0.1.0"
    capabilities = _CAPABILITIES
    # Install from: pip install "git+https://github.com/OSU-NLP-Group/HippoRAG.git"
    # or: pip install hipporag  (if a PyPI release is available)
    required_dependencies = ("hipporag",)
    install_hint = (
        'pip install "git+https://github.com/OSU-NLP-Group/HippoRAG.git"'
        "  # or: pip install hipporag"
    )

    async def _build_native_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        from eidetic.plugins.hipporag._backend import HippoRAGBackend

        return await HippoRAGBackend.create(config)


def get_plugin() -> HippoRAGPlugin:
    return HippoRAGPlugin()
