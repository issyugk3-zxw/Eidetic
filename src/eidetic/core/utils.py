from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from typing import Iterable


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def find_missing_dependencies(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing
