from __future__ import annotations

import importlib.util

import pytest


if importlib.util.find_spec("langchain_core") is None:
    pytestmark = pytest.mark.skip(reason="langchain_core is not installed")


def test_langchain_memory_roundtrip():
    from eidetic.integrations.langchain import EideticLangChainMemory

    memory = EideticLangChainMemory(
        system="letta",
        config={"plugin_config": {"mode": "mock"}},
        input_key="input",
        top_k=4,
    )

    memory.save_context({"input": "My favorite language is Python."}, {"output": "Noted."})
    memory.save_context({"input": "I live in Tokyo."}, {"output": "Got it."})
    variables = memory.load_memory_variables({"input": "favorite language"})
    assert "Python" in variables["history"]

    memory.clear()
    cleared = memory.load_memory_variables({"input": "favorite language"})
    assert "Python" not in cleared["history"]


def test_langchain_memory_empty_input_can_recall():
    from eidetic.integrations.langchain import EideticLangChainMemory

    memory = EideticLangChainMemory(
        system="letta",
        config={"plugin_config": {"mode": "mock"}},
        input_key="input",
        top_k=4,
        session_tag="empty-input-recall",
    )

    memory.save_context({"input": "My city is Tokyo."}, {"output": "stored"})
    recalled = memory.load_memory_variables({"input": ""})
    assert "Tokyo" in recalled["history"]
