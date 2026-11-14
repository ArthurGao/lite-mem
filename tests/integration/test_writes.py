"""S2 acceptance (design §12): direct writes, supersede and correction, soft delete and `as_of`,
erase, idempotency, one transaction per write, namespace isolation and binding."""

import threading
import uuid
from typing import Any

import pytest

from lite_mem.engine import MAX_FACT_CHARS, MemoryEngine
from lite_mem.errors import Conflict, InvalidRequest, NotFound
from lite_mem.stores.base import Store
from lite_mem.testing import FakeEmbedder
from lite_mem.types import Fact, Message, Scope
from tests.conftest import Clock, count_rows, tx_class

ADA = Scope(user_id="ada")
BOB = Scope(user_id="bob")


def _contents(memories: list[Any]) -> list[str]:
    return [m.content for m in memories]


def test_user_messages_are_stored_as_said(engine: MemoryEngine) -> None:
    events = engine.add(
        [Message("user", " Lives in Berlin "), Message("assistant", "Noted!"), Message("user", "")],
        ADA,
        infer=False,
        metadata={"channel": "chat"},
    )
    assert [(e.op, e.content) for e in events] == [("ADD", "Lives in Berlin")]
    [memory] = engine.list_memories(ADA)
    assert (memory.content, memory.type, memory.metadata) == (
        "Lives in Berlin",
        "semantic",
        {"channel": "chat"},
    )
    assert [h.op for h in engine.history(memory.id)] == ["ADD"]


def test_a_new_profile_value_supersedes_the_old_and_the_past_stays_visible(
    engine: MemoryEngine, clock: Clock
) -> None:
    [first] = engine.remember(Fact("Berlin", type="profile", key="city"), ADA)
    before_move = clock.now
    clock.tick()
    [moved] = engine.remember(Fact("Shanghai", type="profile", key="city"), ADA)
    assert (moved.op, moved.replaced) == ("SUPERSEDE", first.memory_id)

    assert _contents(engine.list_memories(ADA)) == ["Shanghai"]
    assert _contents(engine.list_memories(ADA, as_of=before_move)) == ["Berlin"]
    old = engine.get(first.memory_id)
    assert (old.valid_to, old.superseded_by) == (clock.now, moved.memory_id)
    assert [h.op for h in engine.history(first.memory_id)] == ["ADD", "SUPERSEDE"]
    with pytest.raises(Conflict, match="superseded"):
        engine.update(first.memory_id, "Berlin, Germany")


def test_a_correction_keeps_the_id_and_the_old_text_in_history(engine: MemoryEngine) -> None:
    [e] = engine.remember(Fact("Lives in Berln"), ADA)
    fixed = engine.update(e.memory_id, "Lives in Berlin", reason="typo", produced_by="me:manual")
    assert (fixed.id, engine.get(e.memory_id).content) == (e.memory_id, "Lives in Berlin")
    [_, update] = engine.history(e.memory_id)
    assert (update.op, update.before, update.after, update.reason, update.produced_by) == (
        "CORRECT",
        {"content": "Lives in Berln"},
        {"content": "Lives in Berlin"},
        "typo",
        "me:manual",
    )


def test_delete_retires_and_as_of_still_sees_it(engine: MemoryEngine, clock: Clock) -> None:
    [e] = engine.remember(Fact("Likes green tea"), ADA)
    earlier = clock.now
    clock.tick()
    engine.delete(e.memory_id, reason="user asked")
    assert engine.list_memories(ADA) == []
    assert _contents(engine.list_memories(ADA, as_of=earlier)) == ["Likes green tea"]
    assert engine.get(e.memory_id).valid_to == clock.now
    with pytest.raises(Conflict):
        engine.delete(e.memory_id)


def test_erase_removes_everything_about_one_user_and_nothing_else(
    engine: MemoryEngine, store: Store
) -> None:
    engine.remember(Fact("Berlin", type="profile", key="city"), ADA, idempotency_key="k1")
    engine.remember(Fact("Shanghai", type="profile", key="city"), ADA)
    engine.remember(Fact("Likes tea"), BOB, idempotency_key="k2")
    assert engine.erase("ada") == 2

    for table in ("memories", "memory_events", "idempotency"):
        assert count_rows(store, table, "ada") == 0, table
    assert (count_rows(store, "memories", "bob"), count_rows(store, "idempotency", "bob")) == (1, 1)


def test_the_same_idempotency_key_writes_once(engine: MemoryEngine) -> None:
    first = engine.add([Message("user", "Lives in Berlin")], ADA, infer=False, idempotency_key="e1")
    again = engine.add([Message("user", "Lives in Berlin")], ADA, infer=False, idempotency_key="e1")
    assert again == first
    assert len(engine.list_memories(ADA)) == 1
    engine.add([Message("user", "Likes tea")], ADA, infer=False, idempotency_key="e2")
    assert len(engine.list_memories(ADA)) == 2


def test_concurrent_writes_with_one_key_write_once(store: Store, clock: Clock) -> None:
    engines = [MemoryEngine(store, "app", FakeEmbedder(), clock=clock) for _ in range(4)]
    results: list[Any] = []

    def write(e: MemoryEngine) -> None:
        results.append(e.add([Message("user", "Hi")], ADA, infer=False, idempotency_key="same"))

    threads = [threading.Thread(target=write, args=(e,)) for e in engines]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(engines[0].list_memories(ADA)) == 1
    assert all(r == results[0] for r in results)


def test_a_write_that_fails_halfway_leaves_nothing(
    store: Store, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = MemoryEngine(store, "app", FakeEmbedder(), clock=clock)
    calls = {"n": 0}
    tx = tx_class(store)
    original = tx.add_event

    def flaky(self: Any, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        original(self, *args, **kwargs)

    monkeypatch.setattr(tx, "add_event", flaky)
    with pytest.raises(RuntimeError):
        engine.add(
            [Message("user", "a"), Message("user", "b")], ADA, infer=False, idempotency_key="x"
        )
    monkeypatch.undo()
    assert engine.list_memories(ADA) == []
    # the key was not kept either: the retry writes
    assert len(engine.add([Message("user", "a")], ADA, infer=False, idempotency_key="x")) == 1


def test_namespaces_do_not_see_each_other(store: Store, clock: Clock) -> None:
    app = MemoryEngine(store, "app", FakeEmbedder(), clock=clock)
    other = MemoryEngine(store, "other/tenant", FakeEmbedder(), clock=clock)
    [e] = app.remember(Fact("Lives in Berlin"), ADA)
    with pytest.raises(NotFound):
        other.get(e.memory_id)
    with pytest.raises(NotFound):
        other.update(e.memory_id, "x")
    with pytest.raises(NotFound):
        other.history(e.memory_id)
    assert other.list_memories(ADA) == []
    assert other.erase("ada") == 0 and len(app.list_memories(ADA)) == 1


def test_a_namespace_is_bound_to_its_embedding_model(store: Store, clock: Clock) -> None:
    MemoryEngine(store, "app", FakeEmbedder(dim=64), clock=clock)
    with pytest.raises(Conflict, match="dim 64"):
        MemoryEngine(store, "app", FakeEmbedder(dim=32), clock=clock)
    with pytest.raises(Conflict, match="embedding_model"):
        MemoryEngine(store, "app", FakeEmbedder(model="another"), clock=clock)


def test_an_invalid_namespace_is_refused(store: Store) -> None:
    with pytest.raises(InvalidRequest, match="namespace"):
        MemoryEngine(store, "Bad Name", FakeEmbedder())


@pytest.mark.parametrize(
    ("fact", "scope", "message"),
    [
        (Fact("x", type="profile"), ADA, "key"),
        (Fact("x", key="city"), ADA, "key"),
        (Fact("x", type="profile", key="city"), Scope("ada", session_id="s1"), "session"),
        (Fact("x", importance=2), ADA, "importance"),
        (Fact("   "), ADA, "content"),
        (Fact("x" * 2001), ADA, "2000"),
    ],
)
def test_bad_facts_are_refused(
    engine: MemoryEngine, fact: Fact, scope: Scope, message: str
) -> None:
    with pytest.raises(InvalidRequest, match=message):
        engine.remember(fact, scope)


def test_scopes_shared_memories_and_sessions(engine: MemoryEngine) -> None:
    engine.remember(Fact("Shop opens at 9"), Scope())  # shared in the namespace
    engine.remember(Fact("Ada: in session 1"), Scope("ada", session_id="s1"))
    engine.remember(Fact("Ada: always"), ADA)
    engine.remember(Fact("Bob's"), BOB)
    assert _contents(engine.list_memories(ADA)) == ["Ada: in session 1", "Ada: always"]
    assert _contents(engine.list_memories(Scope("ada", session_id="s1"))) == ["Ada: in session 1"]
    assert "Shop opens at 9" in _contents(engine.list_memories(ADA, include_shared=True))
    assert "Bob's" not in _contents(engine.list_memories(ADA, include_shared=True))


def test_unknown_memory_is_not_found(engine: MemoryEngine) -> None:
    with pytest.raises(NotFound):
        engine.get(uuid.uuid4())


def test_an_idempotency_key_never_replays_another_users_write(engine: MemoryEngine) -> None:
    engine.add([Message("user", "Ada's secret plan")], ADA, infer=False, idempotency_key="ep-1")
    with pytest.raises(Conflict, match="idempotency key"):
        engine.add([Message("user", "Bob's note")], BOB, infer=False, idempotency_key="ep-1")
    assert engine.list_memories(BOB) == []


def test_concurrent_first_values_of_one_profile_key_leave_one_current(
    store: Store, clock: Clock
) -> None:
    engines = [MemoryEngine(store, "app", FakeEmbedder(), clock=clock) for _ in range(4)]
    errors: list[BaseException] = []

    def write(e: MemoryEngine, city: str) -> None:
        try:
            e.remember(Fact(city, type="profile", key="city"), ADA)
        except BaseException as exc:  # collected and asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=write, args=(e, c))
        for e, c in zip(engines, ["Berlin", "Paris", "Rome", "Oslo"], strict=True)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(engines[0].list_memories(ADA)) == 1  # one current value, the others superseded


@pytest.mark.parametrize(
    "scope", [Scope(user_id=""), Scope("ada", agent_id=""), Scope("ada", session_id=" ")]
)
def test_empty_scope_ids_are_refused(engine: MemoryEngine, scope: Scope) -> None:
    with pytest.raises(InvalidRequest, match="scope"):
        engine.remember(Fact("x"), scope)


def test_a_message_stored_as_said_is_held_to_a_memory_s_length_up_front(
    engine: MemoryEngine,
) -> None:
    """S11 (LongMemEval): `add` takes messages up to MAX_MESSAGE_CHARS for the extractor, but
    with infer=False each user message is one memory, at most MAX_FACT_CHARS. The write is refused
    whole, before anything is screened or stored, and the error says which limit and why."""
    scope = Scope(user_id="u")
    long = "x" * (MAX_FACT_CHARS + 1)
    with pytest.raises(InvalidRequest, match=r"infer=false.*one memory.*2000 characters"):
        engine.add([Message("user", "short"), Message("user", long)], scope, infer=False)
    assert engine.list_memories(scope) == []
    # Surrounding white space does not count, as for any memory; assistant turns are not stored
    padded = "  " + "y" * MAX_FACT_CHARS + "\n"
    events = engine.add([Message("user", padded), Message("assistant", long)], scope, infer=False)
    assert [e.content for e in events] == ["y" * MAX_FACT_CHARS]
