from __future__ import annotations

from dataclasses import dataclass

from eidetic import (
    CompactRequest,
    Document,
    ForgetRequest,
    MemoryEvent,
    MemoryManager,
    RecallQuery,
)


SYSTEMS = ["letta", "graphrag", "lightrag", "hipporag"]


@dataclass
class DemoResult:
    system: str
    recall_hits: int
    forgot_ok: bool
    compact_ok: bool
    health_status: str
    backend_type: str


def run_for_system(system: str) -> DemoResult:
    manager = MemoryManager()
    memory = manager.create(system, config={"plugin_config": {"mode": "mock"}})

    doc = Document(content=f"{system}: unified memory api", tags=["demo", "doc"])
    event = MemoryEvent(content=f"{system}: user likes short answers", tags=["demo", "profile"])

    memory.ingest([doc])
    memory.remember(event)

    recalled = memory.recall(RecallQuery(query="unified memory api", top_k=5))
    recall_hits = len(recalled.items)

    before_forget = memory.recall(RecallQuery(query="short answers", top_k=5))
    had_profile = any(item.id == event.id for item in before_forget.items)
    memory.forget(ForgetRequest(ids=[event.id], hard_delete=True))
    after_forget = memory.recall(RecallQuery(query="short answers", top_k=5))
    removed_profile = all(item.id != event.id for item in after_forget.items)

    compact_op = memory.compact(CompactRequest(strategy="keep_recent", max_items=10))
    compact_ok = compact_op.success and compact_op.operation == "compact"

    health = memory.healthcheck()
    backend = memory.async_handle.backend

    return DemoResult(
        system=system,
        recall_hits=recall_hits,
        forgot_ok=had_profile and removed_profile,
        compact_ok=compact_ok,
        health_status=str(health.get("status")),
        backend_type=type(backend).__name__,
    )


def main() -> None:
    print("Eidetic quick trial (mock mode)")
    print("=" * 60)

    manager = MemoryManager()
    systems = manager.list_systems()
    print("Discovered systems:", ", ".join(item.system for item in systems))
    print("-" * 60)

    results: list[DemoResult] = []
    for system in SYSTEMS:
        result = run_for_system(system)
        results.append(result)
        print(
            f"[{result.system}] "
            f"recall_hits={result.recall_hits}, "
            f"forget_ok={result.forgot_ok}, "
            f"compact_ok={result.compact_ok}, "
            f"health={result.health_status}, "
            f"backend={result.backend_type}"
        )

    print("-" * 60)
    all_ok = all(
        r.recall_hits > 0 and r.forgot_ok and r.compact_ok and r.health_status == "ok"
        for r in results
    )
    print("Overall:", "PASS" if all_ok else "FAIL")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
