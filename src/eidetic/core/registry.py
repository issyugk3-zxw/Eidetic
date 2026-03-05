from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from eidetic.core.errors import PluginNotFoundError
from eidetic.core.models import SystemInfo
from eidetic.core.protocols import MemoryPlugin
from eidetic.core.utils import find_missing_dependencies


ENTRY_POINT_GROUP = "eidetic.memory_plugins"

_BUILTIN_PLUGIN_TARGETS = {
    "letta": "eidetic.plugins.letta:get_plugin",
    "graphrag": "eidetic.plugins.graphrag:get_plugin",
    "lightrag": "eidetic.plugins.lightrag:get_plugin",
    "hipporag": "eidetic.plugins.hipporag:get_plugin",
}


@dataclass
class PluginSpec:
    name: str
    target: str


class PluginRegistry:
    def __init__(self):
        self._plugins: dict[str, MemoryPlugin] = {}
        self._specs: dict[str, PluginSpec] = self._discover_specs()

    def _discover_specs(self) -> dict[str, PluginSpec]:
        specs: dict[str, PluginSpec] = {}
        eps = entry_points()
        selected: list[EntryPoint] = []
        if hasattr(eps, "select"):
            selected = list(eps.select(group=ENTRY_POINT_GROUP))
        else:
            selected = list(eps.get(ENTRY_POINT_GROUP, []))

        for ep in selected:
            specs[ep.name] = PluginSpec(name=ep.name, target=ep.value)

        # Fallback for local source usage when package isn't installed yet.
        for name, target in _BUILTIN_PLUGIN_TARGETS.items():
            specs.setdefault(name, PluginSpec(name=name, target=target))

        return specs

    def list_system_names(self) -> list[str]:
        return sorted(self._specs.keys())

    def has_system(self, system: str) -> bool:
        return system in self._specs

    def get_system_info(self, system: str) -> SystemInfo:
        plugin = self.get_plugin(system)
        missing = find_missing_dependencies(plugin.required_dependencies)
        return SystemInfo(
            system=system,
            plugin_id=plugin.plugin_id,
            version=plugin.version,
            capabilities=sorted(plugin.capabilities),
            required_dependencies=list(plugin.required_dependencies),
            installed=len(missing) == 0,
            install_hint=plugin.install_hint,
        )

    def list_system_info(self) -> list[SystemInfo]:
        return [self.get_system_info(name) for name in self.list_system_names()]

    def get_plugin(self, system: str) -> MemoryPlugin:
        if system in self._plugins:
            return self._plugins[system]
        spec = self._specs.get(system)
        if spec is None:
            raise PluginNotFoundError(system)
        plugin = self._load_plugin(spec)
        self._plugins[system] = plugin
        return plugin

    @staticmethod
    def _load_plugin(spec: PluginSpec) -> MemoryPlugin:
        module_name, attr_name = spec.target.split(":")
        module = import_module(module_name)
        attr: Any = getattr(module, attr_name)
        plugin = attr() if callable(attr) else attr
        return plugin
