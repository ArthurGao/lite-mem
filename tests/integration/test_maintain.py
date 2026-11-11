"""S7 acceptance (design §6, §12): consolidation and forgetting. An expired memory is not
retrieved but `as_of` still sees it; `maintain` records EXPIRE at the expiry time; similar
memories merge into one with MERGE history; nothing is ever deleted."""

import uuid
from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from lite_mem.config import MaintainConfig
from lite_mem.consolidate import LlmConsolidator
from lite_mem.engine import MemoryEngine
from lite_mem.errors import Conflict, InvalidRequest
from lite_mem.stores.base import Store
from lite_mem.testing import FakeEmbedder, FakeLLM, Handler, memory
from lite_mem.types import Fact, Scope
from tests.conftest import Clock, current

ADA = Scope(user_id="ada")
BOB = Scope(user_id="bob")


def test_an_expired_memory_is_not_retrieved_but_as_of_still_sees_it(
    engine: MemoryEngine, clock: Clock
) -> None:
    created = clock.now
    expires = created + timedelta(hours=1)
    [event] = engine.remember(Fact("parked on level 3", type="episodic", expires_at=expires), ADA)
    engine.remember(Fact("parks at the office"), ADA)
    assert engine.search("parked level", ADA)[0].memory.content == "parked on level 3"
    clock.tick(2 * 3600)
    assert [m.content for m in engine.list_memories(ADA)] == ["parks at the office"]
    assert "parked on level 3" not in [h.memory.content for h in engine.search("parked level", ADA)]
    back = engine.list_memories(ADA, as_of=created + timedelta(minutes=30))
    assert "parked on level 3" in [m.content for m in back]
    assert [h.op for h in engine.history(event.memory_id)] == ["ADD"]  # reads change nothing


def test_maintain_records_each_expiry_at_its_time(engine: MemoryEngine, clock: Clock) -> None:
    expires = clock.now + timedelta(hours=1)
    [event] = engine.remember(Fact("parked on level 3", type="episodic", expires_at=expires), ADA)
    clock.tick(5 * 3600)
    report = engine.maintain()
    assert report.expired == 1
    memory = engine.get(event.memory_id)
    assert memory.valid_to == expires  # when it stopped holding, not when maintain ran
    assert [h.op for h in engine.history(event.memory_id)] == ["ADD", "EXPIRE"]
    assert engine.maintain().expired == 0  # idempotent


def test_a_memory_already_expired_is_refused_when_written(
    engine: MemoryEngine, clock: Clock
) -> None:
    """Found by the CLI test: expires_at before valid_from made maintain violate the
    valid_to >= valid_from check and fail the whole run. Such a memory would never hold."""
    with pytest.raises(InvalidRequest, match="already past"):
        engine.remember(Fact("old news", expires_at=clock.now - timedelta(days=1)), ADA)


def test_a_memory_past_its_expiry_cannot_be_changed(engine: MemoryEngine, clock: Clock) -> None:
    [event] = engine.remember(
        Fact("parked on level 3", expires_at=clock.now + timedelta(hours=1)), ADA
    )
    clock.tick(2 * 3600)
    with pytest.raises(Conflict, match="expired"):
        engine.update(event.memory_id, "parked on level 4")


def test_a_new_profile_value_after_the_old_one_expired_records_the_expiry(
    engine: MemoryEngine, clock: Clock
) -> None:
    soon = clock.now + timedelta(hours=1)
    [old] = engine.remember(Fact("Wellington", type="profile", key="city", expires_at=soon), ADA)
    clock.tick(2 * 3600)
    events = engine.remember(Fact("Auckland", type="profile", key="city"), ADA)
    assert [e.op for e in events] == ["EXPIRE", "ADD"]  # it ran out; nothing superseded it
    assert [h.op for h in engine.history(old.memory_id)] == ["ADD", "EXPIRE"]
    assert engine.get(old.memory_id).valid_to == soon


def _maintaining(
    store: Store, clock: Clock, *, consolidate: Handler | None = None, **config: Any
) -> tuple[MemoryEngine, FakeLLM]:
    llm = FakeLLM(consolidate=consolidate or _same)
    engine = MemoryEngine(
        store, "app", FakeEmbedder(), consolidator=LlmConsolidator(llm),
        maintenance=MaintainConfig(**{"merge_threshold": 0.7, **config}), clock=clock,
    )  # fmt: skip
    return engine, llm


def _same(payload: dict[str, Any]) -> dict[str, Any]:
    """A `consolidate` merge handler: the facts are one; keep the longest wording."""
    longest = max((f["fact"] for f in payload["FACTS"]), key=len)
    return {"same": True, "content": longest, "reason": "same fact"}


def test_similar_memories_merge_into_one_with_merge_history(store: Store, clock: Clock) -> None:
    engine, llm = _maintaining(store, clock)
    [a] = engine.remember(Fact("likes green tea", importance=0.4), ADA)
    [b] = engine.remember(Fact("likes green tea every morning", importance=0.7), ADA)
    engine.remember(Fact("works as a nurse"), ADA)
    clock.tick()
    report = engine.maintain()
    assert report.merged == 2 and len(llm.calls) == 1
    assert current(engine) == ["likes green tea every morning", "works as a nurse"]
    merged = next(m for m in engine.list_memories(ADA) if "tea" in m.content)
    assert merged.id not in (a.memory_id, b.memory_id) and merged.importance == 0.7
    for old in (a.memory_id, b.memory_id):
        closed = engine.get(old)
        assert closed.superseded_by == merged.id and closed.valid_to == clock.now
        assert [h.op for h in engine.history(old)] == ["ADD", "MERGE"]
    [made] = engine.history(merged.id)
    assert made.op == "MERGE" and made.produced_by == "fake:fake-llm"
    assert set(made.after["merged_from"]) == {str(a.memory_id), str(b.memory_id)}
    # the events a mirror (llm-steward) maps: one per memory closed, into the merged one
    assert {(e.op, e.memory_id, e.replaced) for e in report.events} == {
        ("MERGE", merged.id, a.memory_id), ("MERGE", merged.id, b.memory_id)
    }  # fmt: skip
    assert engine.maintain().merged == 0 and len(llm.calls) == 1  # nothing left to merge


def test_the_model_can_say_they_are_different_facts(store: Store, clock: Clock) -> None:
    def different(payload: dict[str, Any]) -> dict[str, Any]:
        return {"same": False, "content": None, "reason": "two cities"}

    engine, llm = _maintaining(store, clock, consolidate=different, merge_threshold=0.6)
    engine.remember(Fact("lives in Auckland"), ADA)
    engine.remember(Fact("lives in Wellington"), ADA)
    assert engine.maintain().merged == 0 and len(llm.calls) == 1
    assert current(engine) == ["lives in Auckland", "lives in Wellington"]


def test_only_one_scopes_semantic_memories_merge(store: Store, clock: Clock) -> None:
    engine, llm = _maintaining(store, clock)
    engine.remember(Fact("likes green tea"), ADA)
    engine.remember(Fact("likes green tea"), BOB)  # another person
    engine.remember(Fact("drank green tea", type="episodic"), ADA)  # not a semantic fact
    engine.remember(Fact("green tea", type="profile", key="drink"), ADA)
    assert engine.maintain().merged == 0 and llm.calls == []


def test_a_merged_fact_is_screened_like_any_derived_fact(store: Store, clock: Clock) -> None:
    def leaky(payload: dict[str, Any]) -> dict[str, Any]:
        return {"same": True, "content": "likes green tea, PIN 482915", "reason": "r"}

    engine, _ = _maintaining(store, clock, consolidate=leaky)
    engine.remember(Fact("likes green tea"), ADA)
    engine.remember(Fact("likes green tea a lot"), ADA)
    assert engine.maintain().merged == 0  # the merged wording would hold a secret: not written
    assert current(engine) == ["likes green tea", "likes green tea a lot"]


def test_old_episodes_are_summarised_when_enabled(store: Store, clock: Clock) -> None:
    def summary(payload: dict[str, Any]) -> dict[str, Any]:
        refunds = [e["id"] for e in payload["EPISODES"] if "refund" in e["fact"]]
        return {"facts": [{"content": "asks about refunds every Friday", "sources": refunds},
                          {"content": "not a source", "sources": ["nonsense"]}]}  # fmt: skip

    engine, llm = _maintaining(
        store, clock, consolidate=summary, merge_threshold=None, summarise_after_days=7,
        summarise_min=2,
    )  # fmt: skip
    for text in ("asked about the refund on Friday", "asked again about the refund on Friday"):
        engine.remember(Fact(text, type="episodic"), ADA)
    engine.remember(Fact("booked a table", type="episodic"), ADA)
    clock.tick(3600)
    assert engine.maintain().summarised == 0 and llm.calls == []  # not old enough yet
    clock.tick(8 * 86400)
    engine.remember(Fact("asked about the refund today", type="episodic"), ADA)  # recent: kept
    report = engine.maintain()
    assert report.summarised == 2
    current = {m.content: m.type for m in engine.list_memories(ADA)}
    assert current == {
        "asks about refunds every Friday": "semantic",
        "booked a table": "episodic",
        "asked about the refund today": "episodic",
    }


def test_idle_low_weight_episodes_are_forgotten_when_enabled(store: Store, clock: Clock) -> None:
    engine, _ = _maintaining(
        store, clock, merge_threshold=None, forget_below=0.3, forget_idle_days=30
    )
    [faint] = engine.remember(Fact("glanced at a menu", type="episodic", importance=0.1), ADA)
    engine.remember(Fact("got married", type="episodic", importance=1.0), ADA)
    [used] = engine.remember(Fact("looked at a menu", type="episodic", importance=0.1), ADA)
    engine.remember(Fact("glanced at the sky", importance=0.1), ADA)  # semantic: never decays out
    clock.tick(60 * 86400)
    engine.search("looked at a menu", ADA, top_k=1, types=["episodic"])  # retrieved: not idle
    report = engine.maintain()
    assert report.forgotten == 1
    assert engine.get(faint.memory_id).valid_to == clock.now
    assert [h.op for h in engine.history(faint.memory_id)] == ["ADD", "EXPIRE"]
    assert engine.get(used.memory_id).current
    assert len(engine.list_memories(ADA)) == 3


def test_forgetting_and_summarising_are_off_by_default() -> None:
    config = MaintainConfig()
    assert (config.summarise_after_days, config.forget_below) == (None, None)
    assert config.merge_threshold == 0.92


def test_a_group_that_changed_during_the_model_call_is_skipped(store: Store, clock: Clock) -> None:
    holder: dict[str, MemoryEngine] = {}

    def meanwhile(payload: dict[str, Any]) -> dict[str, Any]:
        holder["engine"].delete(uuid.UUID(payload["FACTS"][0]["id"]), reason="user removed it")
        return _same(payload)

    engine, _ = _maintaining(store, clock, consolidate=meanwhile)
    holder["engine"] = engine
    engine.remember(Fact("likes green tea"), ADA)
    engine.remember(Fact("likes green tea every morning"), ADA)
    assert engine.maintain().merged == 0
    assert current(engine) == ["likes green tea every morning"]  # untouched, not merged


def test_two_runs_at_once_merge_a_group_once(store: Store, clock: Clock) -> None:
    holder: dict[str, Any] = {"inner": None}

    def racing(payload: dict[str, Any]) -> dict[str, Any]:
        if holder["inner"] is None:  # the first run's model call: a second run goes meanwhile
            holder["inner"] = "running"
            holder["inner"] = holder["engine"].maintain()
        return _same(payload)

    engine, _ = _maintaining(store, clock, consolidate=racing)
    holder["engine"] = engine
    engine.remember(Fact("likes green tea"), ADA)
    engine.remember(Fact("likes green tea every morning"), ADA)
    outer = engine.maintain()
    assert (holder["inner"].merged, outer.merged) == (2, 0)
    assert current(engine) == ["likes green tea every morning"]  # one merged memory, not two


def test_a_merge_never_crosses_scopes(store: Store, clock: Clock) -> None:
    """Defence in depth: the groups come from one scope's query, but the write checks again."""
    engine, _ = _maintaining(store, clock)
    [bobs] = engine.remember(Fact("likes green tea"), BOB)
    fact = Fact("likes green tea")
    steps = engine._merging(ADA, fact, [engine.get(bobs.memory_id)], "r", None)
    assert engine._commit_merges(ADA, steps) == []
    assert engine.get(bobs.memory_id).current and current(engine) == []


def test_types_merge_only_with_their_own_kind(store: Store, clock: Clock) -> None:
    engine, llm = _maintaining(store, clock)
    engine.remember(Fact("likes green tea"), ADA)
    engine.remember(Fact("likes green tea", type="procedural"), ADA)
    assert engine.maintain().merged == 0 and llm.calls == []
    with pytest.raises(ValidationError, match="profile"):
        MaintainConfig(merge_types=("semantic", "profile"))


def test_a_merged_memory_expires_only_when_all_its_sources_would(
    store: Store, clock: Clock
) -> None:
    engine, _ = _maintaining(store, clock)
    later = clock.now + timedelta(days=9)
    engine.remember(Fact("likes green tea", expires_at=clock.now + timedelta(days=3)), ADA)
    engine.remember(Fact("likes green tea a lot", expires_at=later), ADA)
    engine.remember(Fact("works as a nurse", expires_at=later), ADA)
    engine.remember(Fact("works as a nurse now"), ADA)
    engine.maintain()
    ends = {m.content: m.expires_at for m in engine.list_memories(ADA)}
    assert ends == {"likes green tea a lot": later, "works as a nurse now": None}


def test_summary_sources_are_given_ids_each_used_once() -> None:
    episodes = [memory(f"episode {n}", type="episodic") for n in range(3)]
    a, b, c = (str(e.id) for e in episodes)
    llm = FakeLLM(consolidate=lambda p: {"facts": [
        {"content": "first", "sources": [a, b, a]},
        {"content": "second", "sources": [b, c, "not-an-id"]},
        {"content": "third", "sources": [a]},
        {"content": "  ", "sources": [c]},
    ]})  # fmt: skip
    summaries = LlmConsolidator(llm).summarise(episodes)
    assert [(s.content, s.sources) for s in summaries] == [
        ("first", (episodes[0].id, episodes[1].id)),
        ("second", (episodes[2].id,)),
    ]
