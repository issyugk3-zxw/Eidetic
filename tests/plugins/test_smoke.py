from __future__ import annotations

import pytest

from eidetic import MemoryManager


@pytest.mark.parametrize("system", ["letta", "graphrag", "lightrag", "hipporag"])
def test_plugin_smoke(system: str):
    manager = MemoryManager()
    handle = manager.create(system, {"plugin_config": {"mode": "mock"}})
    health = handle.healthcheck()
    assert health["status"] == "ok"
    assert health["system"] == system


def test_list_systems_contains_mvp_plugins():
    manager = MemoryManager()
    systems = {item.system for item in manager.list_systems()}
    assert {"letta", "graphrag", "lightrag", "hipporag"}.issubset(systems)
