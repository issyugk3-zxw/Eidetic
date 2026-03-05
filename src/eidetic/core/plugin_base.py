from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

from eidetic.core.errors import DependencyMissingError
from eidetic.core.in_memory_backend import InMemorySemanticBackend
from eidetic.core.protocols import AsyncMemoryBackend
from eidetic.core.utils import find_missing_dependencies

_VALID_MODES = {"auto", "native", "persistent", "mock"}


class BasePlugin(ABC):
    """
    Base class for all Eidetic memory plugins.

    Subclasses **must** override ``_build_native_backend`` to connect to
    the real backend library.  The mode selection flow is:

    - ``"mock"``       → in-memory (no external deps, data lost on exit)
    - ``"persistent"`` → SQLite (no external deps, data survives restarts)
    - ``"native"``     → real backend; raises DependencyMissingError if
                         required packages are absent
    - ``"auto"``       → try native; fall back to persistent with a warning
                         if required packages are missing
    """

    plugin_id: str
    version: str
    capabilities: set[str]
    required_dependencies: tuple[str, ...]
    install_hint: str

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def create(self, config: dict[str, Any] | None = None) -> AsyncMemoryBackend:
        config = config or {}
        plugin_config = config.get("plugin_config", {})
        mode = plugin_config.get("mode", "auto")

        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid plugin mode '{mode}' for '{self.plugin_id}'. "
                f"Expected one of: {', '.join(sorted(_VALID_MODES))}."
            )

        if mode == "mock":
            return self._build_mock_backend(config)

        if mode == "persistent":
            return self._build_persistent_backend(config)

        missing = find_missing_dependencies(self.required_dependencies)

        if mode == "native":
            if missing:
                raise DependencyMissingError(self.plugin_id, missing, self.install_hint)
            return await self._build_native_backend(config)

        # mode == "auto"
        if missing:
            warnings.warn(
                f"[eidetic] Native backend '{self.plugin_id}' requires "
                f"{missing}. Falling back to SQLite persistent backend. "
                f"Install with: {self.install_hint}",
                UserWarning,
                stacklevel=3,
            )
            return self._build_persistent_backend(config)

        return await self._build_native_backend(config)

    # ------------------------------------------------------------------
    # Backend factories
    # ------------------------------------------------------------------

    def _build_mock_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        """In-memory backend for tests and quick prototyping."""
        return InMemorySemanticBackend(system=self.plugin_id, native=False)

    def _build_persistent_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        """SQLite-backed persistent backend, zero extra dependencies."""
        from eidetic.core.sqlite_backend import SqliteBackend

        plugin_config = config.get("plugin_config", {})
        db_path = plugin_config.get("db_path", f"eidetic_{self.plugin_id}.db")
        return SqliteBackend(system=self.plugin_id, db_path=db_path)

    @abstractmethod
    async def _build_native_backend(self, config: dict[str, Any]) -> AsyncMemoryBackend:
        """
        Build the real backend using the plugin's native library.

        This method is called only when all ``required_dependencies`` are
        present.  Subclasses must override this and return a fully
        initialised ``AsyncMemoryBackend``.

        ``config`` is the raw dict passed by the caller.  Plugin-specific
        options should live under ``config["plugin_config"]``.
        """
        ...
