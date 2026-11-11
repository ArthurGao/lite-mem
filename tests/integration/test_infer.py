"""S4 acceptance (design §12): add(infer=true) — the extractor proposes, each fact is resolved
against its related memories of exactly the same scope, the batch commits in one transaction, a
retry with the same key calls no model, every model-made change records provider:model."""

import uuid
from typing import Any

import pytest

from lite_mem.engine import MemoryEngine
from lite_mem.extract import LlmExtractor
from lite_mem.resolve import LlmResolver
from lite_mem.stores.pg import PgStore
from lite_mem.testing import FakeEmbedder, FakeLLM, Handler, update_when, user_lines
from lite_mem.types import Message, Scope
from tests.conftest import Clock

ADA = Scope(user_id="ada")
BOB = Scope(user_id="bob")


def _engine(store: PgStore, clock: Clock, resolve: Handler) -> tuple[MemoryEngine, FakeLLM]:
    llm = FakeLLM(extract=user_lines, resolve=resolve)
    engine = MemoryEngine(
        store, "app", FakeEmbedder(), extractor=LlmExtractor(llm), resolver=LlmResolver(llm),
        clock=clock,
    )  # fmt: skip
    return engine, llm


def _say(engine: MemoryEngine, scope: Scope, *texts: str, **kw: Any) -> Any:
    return engine.add([Message("user", t) for t in texts], scope, **kw)


def _current(engine: MemoryEngine, scope: Scope = ADA) -> list[str]:
    return sorted(m.content for m in engine.list_memories(scope))


def test_beijing_then_shanghai_supersedes(store: PgStore, clock: Clock) -> None:
    engine, llm = _engine(store, clock, update_when("搬"))
    [first] = _say(engine, ADA, "我住北京")
    assert (first.op, first.produced_by) == ("ADD", "fake:fake-llm")
    then = clock.now
    clock.tick()
    [moved] = _say(engine, ADA, "我搬到上海了")
    assert (moved.op, moved.replaced, moved.produced_by) == (
        "SUPERSEDE",
        first.memory_id,
        "fake:fake-llm",
    )
    assert _current(engine) == ["我搬到上海了"]
    assert [m.content for m in engine.list_memories(ADA, as_of=then)] == ["我住北京"]
    assert [h.op for h in engine.history(first.memory_id)] == ["ADD", "SUPERSEDE"]
    resolve_calls = [p for r, p in llm.calls if r == "resolve"]
    assert [e["fact"] for e in resolve_calls[0]["EXISTING"]] == ["我住北京"]


def test_a_duplicate_is_not_written_again(store: PgStore, clock: Clock) -> None:
    engine, _ = _engine(
        store, clock, lambda p: {"action": "NOOP", "target_id": None, "reason": "same"}
    )
    _say(engine, ADA, "我喜欢绿茶")  # nothing related yet: added without asking
    assert _say(engine, ADA, "我喜欢绿茶") == []
    assert _current(engine) == ["我喜欢绿茶"]


def test_a_target_outside_the_candidates_is_added_instead(store: PgStore, clock: Clock) -> None:
    stray = str(uuid.uuid4())
    engine, _ = _engine(
        store, clock, lambda p: {"action": "UPDATE", "target_id": stray, "reason": "r"}
    )
    _say(engine, ADA, "Lives in Berlin")
    [e] = _say(engine, ADA, "Lives in Paris")
    assert e.op == "ADD" and _current(engine) == ["Lives in Berlin", "Lives in Paris"]


def test_delete_retires_the_old_fact_and_keeps_the_new_one(store: PgStore, clock: Clock) -> None:
    def no_more(payload: dict[str, Any]) -> dict[str, Any]:
        return {"action": "DELETE", "target_id": payload["EXISTING"][0]["id"], "reason": "gone"}

    engine, _ = _engine(store, clock, no_more)
    _say(engine, ADA, "Has a cat")
    events = _say(engine, ADA, "No longer has a cat")
    assert [e.op for e in events] == ["DELETE", "ADD"]
    assert events[0].reason == "gone" and events[0].produced_by == "fake:fake-llm"
    assert _current(engine) == ["No longer has a cat"]


def test_related_memories_come_from_exactly_this_scope(store: PgStore, clock: Clock) -> None:
    engine, llm = _engine(store, clock, update_when("搬"))
    _say(engine, BOB, "我住北京")
    _say(engine, Scope(), "公司在北京")  # shared: visible to Ada, but not hers to supersede
    _say(engine, ADA, "我搬到上海了")
    assert [p for r, p in llm.calls if r == "resolve"] == []  # nothing of hers to compare with
    assert _current(engine, BOB) == ["我住北京"]


def test_a_retry_with_the_same_key_calls_no_model(store: PgStore, clock: Clock) -> None:
    engine, llm = _engine(store, clock, update_when("搬"))
    first = _say(engine, ADA, "我住北京", idempotency_key="episode-1")
    calls = len(llm.calls)
    assert _say(engine, ADA, "我住北京", idempotency_key="episode-1") == first
    assert len(llm.calls) == calls and _current(engine) == ["我住北京"]


def test_the_batch_is_one_transaction(
    store: PgStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lite_mem.stores import pg

    engine, _ = _engine(store, clock, update_when("搬"))
    _say(engine, ADA, "我住北京")
    original, calls = pg.PgTx.insert_memory, {"n": 0}

    def flaky(self: pg.PgTx, *args: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        original(self, *args)

    monkeypatch.setattr(pg.PgTx, "insert_memory", flaky)
    with pytest.raises(RuntimeError):
        _say(engine, ADA, "我搬到上海了", "我喜欢绿茶")  # a supersede, then a failing add
    monkeypatch.undo()
    assert _current(engine) == ["我住北京"]  # the supersede rolled back with it


def test_a_target_changed_meanwhile_is_added_instead(store: PgStore, clock: Clock) -> None:
    """The memory the resolver chose was retired between the lookup and the write."""
    holder: dict[str, MemoryEngine] = {}

    def update_then_retire(payload: dict[str, Any]) -> dict[str, Any]:
        target = payload["EXISTING"][0]["id"]
        holder["engine"].delete(uuid.UUID(target))  # a concurrent change, as the model thinks
        return {"action": "UPDATE", "target_id": target, "reason": "moved"}

    engine, _ = _engine(store, clock, update_then_retire)
    holder["engine"] = engine
    _say(engine, ADA, "Lives in Berlin")
    [e] = _say(engine, ADA, "Lives in Paris")
    assert e.op == "ADD" and _current(engine) == ["Lives in Paris"]


def test_without_an_extractor_infer_is_a_config_error(engine: MemoryEngine) -> None:
    from lite_mem.errors import ConfigError

    with pytest.raises(ConfigError, match="extractor"):
        _say(engine, ADA, "x")


@pytest.mark.ollama
def test_a_real_round_trip_with_local_models(store: PgStore, clock: Clock) -> None:
    """qwen3:8b extracts and resolves, bge-m3 embeds — all on the local Ollama."""
    from lite_mem.config import EmbeddingConfig, LLMConfig
    from lite_mem.embedding import OllamaEmbedder
    from lite_mem.llm import RoutedLLM

    llm = RoutedLLM(LLMConfig())
    engine = MemoryEngine(
        store, "real", OllamaEmbedder(EmbeddingConfig()), extractor=LlmExtractor(llm),
        resolver=LlmResolver(llm), clock=clock,
    )  # fmt: skip
    first = engine.add([Message("user", "你好，我叫小王，我住在北京，平时喜欢喝绿茶。")], ADA)
    assert first and all(e.produced_by and e.produced_by.startswith("openai:qwen3") for e in first)
    clock.tick()
    engine.add([Message("user", "跟你说一下，我上个月搬到上海了。")], ADA)
    current = " ".join(_current(engine))
    # the model may answer in either language (measured in S5): assert the structure only
    assert "北京" not in current and "Beijing" not in current  # superseded
    assert "上海" in current or "Shanghai" in current
    assert "绿茶" in current or "green tea" in current.lower()  # unrelated facts stay


def test_facts_of_one_batch_are_resolved_against_each_other(store: PgStore, clock: Clock) -> None:
    """One message, two wordings of one change: the second sees the first (pending in the batch)
    and is a duplicate of it, not a second current fact (seen with qwen3:8b)."""

    def resolve(payload: dict[str, Any]) -> dict[str, Any]:
        existing = payload["EXISTING"]
        if "上海" in payload["NEW"]:
            shanghai = [e for e in existing if "上海" in e["fact"]]
            if shanghai:
                return {"action": "NOOP", "target_id": None, "reason": "same move"}
            beijing = [e for e in existing if "北京" in e["fact"]]
            if beijing:
                return {"action": "UPDATE", "target_id": beijing[0]["id"], "reason": "moved"}
        return {"action": "ADD", "target_id": None, "reason": "new"}

    engine, _ = _engine(store, clock, resolve)
    _say(engine, ADA, "住在北京")
    events = _say(engine, ADA, "住在上海", "上个月搬到上海")
    assert [e.op for e in events] == ["SUPERSEDE"]
    assert _current(engine) == ["住在上海"]


def test_a_later_fact_can_replace_an_earlier_one_of_its_batch(store: PgStore, clock: Clock) -> None:
    """The later wording wins and inherits what the earlier one replaced."""

    def resolve(payload: dict[str, Any]) -> dict[str, Any]:
        existing = payload["EXISTING"]
        pending = [e for e in existing if e["fact"] == "搬到上海"]
        if payload["NEW"] == "上个月搬到了上海浦东" and pending:
            return {"action": "UPDATE", "target_id": pending[0]["id"], "reason": "more precise"}
        beijing = [e for e in existing if "北京" in e["fact"]]
        if beijing:
            return {"action": "UPDATE", "target_id": beijing[0]["id"], "reason": "moved"}
        return {"action": "ADD", "target_id": None, "reason": "new"}

    engine, _ = _engine(store, clock, resolve)
    [first] = _say(engine, ADA, "住在北京")
    [event] = _say(engine, ADA, "搬到上海", "上个月搬到了上海浦东")
    assert (event.op, event.content, event.replaced) == (
        "SUPERSEDE",
        "上个月搬到了上海浦东",
        first.memory_id,
    )
    assert _current(engine) == ["上个月搬到了上海浦东"]


def test_a_resolver_cannot_reach_another_users_memory(store: PgStore, clock: Clock) -> None:
    """Whatever a resolver answers, only its candidates can be superseded or deleted."""
    holder: dict[str, str] = {}

    def aim_at_bob(payload: dict[str, Any]) -> dict[str, Any]:
        return {"action": "UPDATE", "target_id": holder["bob"], "reason": "wrong user"}

    engine, _ = _engine(store, clock, aim_at_bob)
    [bobs] = _say(engine, BOB, "Lives in Berlin")
    holder["bob"] = str(bobs.memory_id)
    _say(engine, ADA, "Lives in Rome")  # nothing of hers yet: resolved without asking
    [event] = _say(engine, ADA, "Lives in Paris")
    assert event.op == "ADD" and "not among the candidates" in (event.reason or "")
    assert _current(engine, BOB) == ["Lives in Berlin"]  # untouched


def test_an_update_keeps_the_part_of_a_compound_fact_that_still_holds(
    store: PgStore, clock: Clock
) -> None:
    """S6 acceptance run: superseding "lives in Wellington and works as a nurse" by a move lost
    the job. The resolver names what still holds; code keeps it as its own memory."""
    engine, _ = _engine(store, clock, update_when("moved", remainder="works as a nurse"))
    [first] = _say(engine, ADA, "lives in Wellington and works as a nurse")
    events = _say(engine, ADA, "moved to Auckland")
    assert _current(engine) == ["moved to Auckland", "works as a nurse"]
    by_op = {e.op: e for e in events}
    assert by_op["SUPERSEDE"].replaced == first.memory_id
    kept = by_op["ADD"]
    assert kept.content == "works as a nurse" and kept.produced_by == "fake:fake-llm"
    assert str(first.memory_id) in (kept.reason or "")


def test_a_remainder_is_screened_like_any_derived_fact(store: PgStore, clock: Clock) -> None:
    engine, _ = _engine(store, clock, update_when("moved", remainder="her PIN is 482915"))
    _say(engine, ADA, "lives in Wellington")
    _say(engine, ADA, "moved to Auckland")
    assert _current(engine) == ["moved to Auckland"]
