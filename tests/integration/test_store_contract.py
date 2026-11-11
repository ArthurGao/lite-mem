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


def test_a_session_only_memory_is_not_shared_with_other_sessions(
    engine: MemoryEngine,
) -> None:
    """Codex S8: `shared` matched rows with no user and no agent even when they had a session,
    so one session's memory reached every other session."""
    engine.remember(Fact("only in s2"), Scope(session_id="s2"))
    engine.remember(Fact("for everyone"), Scope())
    s1 = Scope(user_id="ada", session_id="s1")
    seen = [m.content for m in engine.list_memories(s1, include_shared=True)]
    assert seen == ["for everyone"]
    assert [h.memory.content for h in engine.search("only in s2", s1)] == ["for everyone"]


def test_secrets_in_metadata_are_screened_like_content(store: Store) -> None:
    """Codex S8: metadata was stored as given, a password in it included."""
    from lite_mem.testing import FakeEmbedder
    from lite_mem.types import Message

    drop = MemoryEngine(store, "app", FakeEmbedder())
    with pytest.raises(InvalidRequest, match="metadata"):
        drop.remember(Fact("wifi set up", metadata={"note": {"pw": "password is tiger-77"}}), ADA)
    with pytest.raises(InvalidRequest, match="metadata"):
        drop.add([Message("user", "hello")], ADA, infer=False, metadata={"k": "PIN 4829"})
    assert drop.list_memories(ADA) == []
    masked = MemoryEngine(store, "app", FakeEmbedder(), sensitive="mask")
    masked.remember(Fact("wifi set up", metadata={"note": "password is tiger-77", "n": 3}), ADA)
    [kept] = masked.list_memories(ADA)
    assert kept.metadata == {"note": "password is [redacted]", "n": 3}


def test_writes_and_lists_are_bounded_the_same_everywhere(engine: MemoryEngine) -> None:
    """Codex S8: no limit on messages per write, on a message's length, or on a list."""
    from lite_mem.engine import MAX_LIST, MAX_MESSAGE_CHARS, MAX_MESSAGES
    from lite_mem.types import Message

    with pytest.raises(InvalidRequest, match="messages"):
        engine.add([Message("user", "x")] * (MAX_MESSAGES + 1), ADA, infer=False)
    with pytest.raises(InvalidRequest, match="characters"):
        engine.add([Message("user", "x" * (MAX_MESSAGE_CHARS + 1))], ADA, infer=False)
    for n in range(5):
        engine.remember(Fact(f"fact {n}"), ADA)
    page = engine.list_memories(ADA, limit=2, offset=1)
    assert [m.content for m in page] == ["fact 1", "fact 2"]
    with pytest.raises(InvalidRequest, match="limit"):
        engine.list_memories(ADA, limit=MAX_LIST + 1)
