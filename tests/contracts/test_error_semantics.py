from __future__ import annotations

import pytest

from eidetic import (
    MemoryEvent,
    MemoryManager,
    RecallQuery,
)
from eidetic.core.errors import (
    BackendOperationError,
    CapabilityNotSupportedError,
    DependencyMissingError,
    PluginNotFoundError,
)


def test_plugin_not_found():
    manager = MemoryManager()
    with pytest.raises(PluginNotFoundError):
        manager.create("unknown-system")


def test_dependency_missing_error():
    manager = MemoryManager()
    plugin = manager._registry.get_plugin("letta")
    original = plugin.required_dependencies
    plugin.required_dependencies = ("definitely_missing_pkg_123456789",)
    try:
        with pytest.raises(DependencyMissingError):
            manager.create("letta", config={"plugin_config": {"mode": "native"}})
    finally:
        plugin.required_dependencies = original


def test_capability_not_supported_error():
    manager = MemoryManager()
    plugin = manager._registry.get_plugin("graphrag")
    original = set(plugin.capabilities)
    plugin.capabilities = {"ingest", "remember", "recall", "forget"}
    try:
        memory = manager.create("graphrag", config={"plugin_config": {"mode": "mock"}})
        with pytest.raises(CapabilityNotSupportedError):
            memory.compact()
    finally:
        plugin.capabilities = original


def test_backend_operation_error():
    manager = MemoryManager()
    memory = manager.create("lightrag", config={"plugin_config": {"mode": "mock"}})

    async def _broken(_query):
        raise RuntimeError("boom")

    memory._async_handle._backend.recall = _broken
    with pytest.raises(BackendOperationError):
        memory.recall(RecallQuery(query="boom"))

    # sanity check still writes fine once error path is covered
    memory.remember(MemoryEvent(content="healthy"))
