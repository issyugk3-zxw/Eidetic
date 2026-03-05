from __future__ import annotations

import pytest

from eidetic import (
    CompactRequest,
    Document,
    ForgetRequest,
    MemoryEvent,
    MemoryManager,
    RecallQuery,
)


SYSTEMS = ["letta", "graphrag", "lightrag", "hipporag"]


@pytest.fixture(params=SYSTEMS)
def memory(request):
    manager = MemoryManager()
    handle = manager.create(
        request.param,
        config={"plugin_config": {"mode": "mock"}},
    )
    return handle


def test_remember_recall_consistency(memory):
    content = "alpha-agent-memory-contract"
    memory.remember(MemoryEvent(content=content, tags=["contract", "remember"]))
    result = memory.recall(RecallQuery(query="alpha-agent-memory-contract", top_k=3))
    assert any(item.content == content for item in result.items)


def test_ingest_recall_retrievability(memory):
    doc = Document(content="beta-ingest-recall-contract", tags=["contract", "ingest"])
    memory.ingest([doc])
    result = memory.recall(RecallQuery(query="beta-ingest-recall-contract", top_k=5))
    assert any(item.id == doc.id for item in result.items)


def test_forget_visibility(memory):
    event = MemoryEvent(content="gamma-forget-contract", tags=["contract", "forget"])
    op = memory.remember(event)
    assert op.processed == 1

    before = memory.recall(RecallQuery(query="gamma-forget-contract", top_k=5))
    assert any(item.id == event.id for item in before.items)

    forget_op = memory.forget(ForgetRequest(ids=[event.id], hard_delete=True))
    assert event.id in forget_op.affected_ids

    after = memory.recall(RecallQuery(query="gamma-forget-contract", top_k=5))
    assert all(item.id != event.id for item in after.items)


def test_compact_callable(memory):
    memory.remember(MemoryEvent(content="delta-compact-1", tags=["contract", "compact"]))
    memory.remember(MemoryEvent(content="delta-compact-2", tags=["contract", "compact"]))
    op = memory.compact(CompactRequest(strategy="keep_recent", max_items=1))
    assert op.operation == "compact"
    assert op.success is True
