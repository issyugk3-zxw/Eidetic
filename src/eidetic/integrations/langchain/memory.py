from __future__ import annotations

import importlib.util
from typing import Any

from eidetic.core.errors import DependencyMissingError
from eidetic.core.manager import MemoryManager
from eidetic.core.models import ForgetRequest, MemoryEvent, RecallQuery


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


LANGCHAIN_AVAILABLE = _has_module("langchain") and _has_module("langchain_core")

_BaseMemory = None
if LANGCHAIN_AVAILABLE:
    try:
        from langchain_core.memory import BaseMemory as _BaseMemory  # type: ignore[attr-defined]
    except Exception:
        _BaseMemory = None


if LANGCHAIN_AVAILABLE and _BaseMemory is not None:
    from pydantic import ConfigDict, Field, PrivateAttr

    class EideticLangChainMemory(_BaseMemory):
        """LangChain memory adapter backed by Eidetic."""

        manager: MemoryManager = Field(default_factory=MemoryManager)
        system: str = "letta"
        config: dict[str, Any] = Field(default_factory=dict)
        memory_key: str = "history"
        input_key: str = "input"
        top_k: int = 5
        session_tag: str = "langchain-session"

        _handle: Any = PrivateAttr(default=None)

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def model_post_init(self, __context: Any) -> None:
            config = self.config or {"plugin_config": {"mode": "mock"}}
            self._handle = self.manager.create(self.system, config=config)

        @property
        def memory_variables(self) -> list[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
            query_text = str(inputs.get(self.input_key, ""))
            result = self._handle.recall(
                RecallQuery(
                    query=query_text,
                    top_k=self.top_k,
                    tags=[self.session_tag],
                    include_documents=False,
                    include_events=True,
                )
            )
            history = "\n".join(item.content for item in result.items)
            return {self.memory_key: history}

        def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
            user_part = str(inputs.get(self.input_key, ""))
            ai_part = str(outputs.get("output", ""))
            content = f"user: {user_part}\nassistant: {ai_part}"
            event = MemoryEvent(content=content, role="system", tags=[self.session_tag])
            self._handle.remember(event)

        def clear(self) -> None:
            self._handle.forget(ForgetRequest(tags=[self.session_tag], hard_delete=True))

elif LANGCHAIN_AVAILABLE:
    class EideticLangChainMemory:
        """LangChain-style memory adapter for newer LangChain versions."""

        def __init__(
            self,
            manager: MemoryManager | None = None,
            system: str = "letta",
            config: dict[str, Any] | None = None,
            memory_key: str = "history",
            input_key: str = "input",
            top_k: int = 5,
            session_tag: str = "langchain-session",
        ):
            self.manager = manager or MemoryManager()
            self.system = system
            self.config = config or {"plugin_config": {"mode": "mock"}}
            self.memory_key = memory_key
            self.input_key = input_key
            self.top_k = top_k
            self.session_tag = session_tag
            self._handle = self.manager.create(self.system, config=self.config)

        @property
        def memory_variables(self) -> list[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
            query_text = str(inputs.get(self.input_key, ""))
            result = self._handle.recall(
                RecallQuery(
                    query=query_text,
                    top_k=self.top_k,
                    tags=[self.session_tag],
                    include_documents=False,
                    include_events=True,
                )
            )
            history = "\n".join(item.content for item in result.items)
            return {self.memory_key: history}

        def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
            user_part = str(inputs.get(self.input_key, ""))
            ai_part = str(outputs.get("output", ""))
            content = f"user: {user_part}\nassistant: {ai_part}"
            event = MemoryEvent(content=content, role="system", tags=[self.session_tag])
            self._handle.remember(event)

        def clear(self) -> None:
            self._handle.forget(ForgetRequest(tags=[self.session_tag], hard_delete=True))

else:
    class EideticLangChainMemory:
        def __init__(self, *args: Any, **kwargs: Any):
            raise DependencyMissingError(
                system="langchain",
                missing=["langchain", "langchain_core"],
                install_hint='pip install "eidetic[langchain]"',
            )
