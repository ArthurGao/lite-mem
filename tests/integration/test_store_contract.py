"""S7b review: rules both stores must apply the same way (the `store` fixture runs each test on
PostgreSQL and on SQLite)."""

from datetime import datetime

import pytest

from lite_mem.engine import MemoryEngine
from lite_mem.errors import InvalidRequest
from lite_mem.stores.base import Filters, Store
from lite_mem.types import Fact, Scope

ADA = Scope(user_id="ada")


def _matching(store: Store, metadata: dict[str, object]) -> list[str]:
    with store.transaction() as tx:
        return [m.content for m in tx.list_memories("app", Filters(ADA, metadata=metadata))]


def test_metadata_filters_are_jsonb_containment(engine: MemoryEngine, store: Store) -> None:
    """Codex S7b: SQLite compared values for equality, PostgreSQL tests containment (`@>`)."""
    engine.remember(Fact("a", metadata={"prefs": {"color": "blue", "size": "L"}}), ADA)
    engine.remember(Fact("b", metadata={"tags": ["x", "y"], "on": True}), ADA)
    engine.remember(Fact("c", metadata={"on": 1}), ADA)
    assert _matching(store, {"prefs": {"color": "blue"}}) == ["a"]  # a nested part is enough
    assert _matching(store, {"tags": ["y"]}) == ["b"]  # one element of the array
    assert _matching(store, {"on": True}) == ["b"]  # true is not 1
    assert _matching(store, {"on": 1}) == ["c"]
    assert _matching(store, {"prefs": {"color": "red"}}) == []


def test_a_time_without_a_zone_is_refused(engine: MemoryEngine) -> None:
    """Codex S7b: a naive time meant the machine's zone in SQLite, the session's in PostgreSQL."""
    naive = datetime(2026, 1, 1, 9, 0)
    with pytest.raises(InvalidRequest, match="timezone"):
        engine.list_memories(ADA, as_of=naive)
    with pytest.raises(InvalidRequest, match="timezone"):
        engine.remember(Fact("parked", expires_at=naive), ADA)
