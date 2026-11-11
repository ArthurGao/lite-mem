"""S5b acceptance (design §12): secrets never become memories. What the engine derives itself
(extracted facts, messages stored as said) is dropped or masked; a fact a caller states explicitly
is refused under `drop`, so the caller learns it was not stored."""

import pytest

from lite_mem.engine import MemoryEngine
from lite_mem.errors import InvalidRequest
from lite_mem.extract import LlmExtractor
from lite_mem.resolve import LlmResolver
from lite_mem.sensitive import Policy
from lite_mem.stores.pg import PgStore
from lite_mem.testing import FakeEmbedder, FakeLLM, update_when, user_lines
from lite_mem.types import Fact, Message, Scope

ADA = Scope(user_id="ada")
SAID = ["住在上海", "我的银行卡密码是 482915", "card 4111 1111 1111 1111 is my visa"]


def _engine(store: PgStore, sensitive: Policy) -> MemoryEngine:
    llm = FakeLLM(extract=user_lines, resolve=update_when("搬"))
    return MemoryEngine(
        store, "app", FakeEmbedder(), extractor=LlmExtractor(llm), resolver=LlmResolver(llm),
        sensitive=sensitive,
    )  # fmt: skip


def _current(engine: MemoryEngine) -> list[str]:
    return sorted(m.content for m in engine.list_memories(ADA))


@pytest.mark.parametrize("infer", [True, False])
def test_drop_keeps_secrets_out_of_derived_memories(store: PgStore, infer: bool) -> None:
    engine = _engine(store, "drop")
    events = engine.add([Message("user", t) for t in SAID], ADA, infer=infer)
    assert _current(engine) == ["住在上海"]
    assert len(events) == 1
    history = [e for m in engine.list_memories(ADA) for e in engine.history(m.id)]
    assert "482915" not in repr(history)


def test_mask_keeps_the_fact_without_the_value(store: PgStore) -> None:
    engine = _engine(store, "mask")
    engine.add([Message("user", t) for t in SAID], ADA)
    assert _current(engine) == [
        "card [redacted] is my visa",
        "住在上海",
        "我的银行卡密码是 [redacted]",
    ]


def test_keep_stores_what_was_said(store: PgStore) -> None:
    engine = _engine(store, "keep")
    engine.add([Message("user", SAID[1])], ADA)
    assert _current(engine) == ["我的银行卡密码是 482915"]


def test_an_explicit_secret_is_refused_under_drop(store: PgStore) -> None:
    engine = _engine(store, "drop")
    with pytest.raises(InvalidRequest, match="credential"):
        engine.remember(Fact("wifi password is tiger-77"), ADA)
    [event] = engine.remember(Fact("wifi is in the hallway"), ADA)
    with pytest.raises(InvalidRequest, match="card"):
        engine.update(event.memory_id, "pays with 4111 1111 1111 1111")
    assert _current(engine) == ["wifi is in the hallway"]


def test_a_reason_never_carries_a_secret_into_history(store: PgStore) -> None:
    engine = _engine(store, "drop")
    [event] = engine.remember(Fact("works in an office"), ADA)
    engine.update(event.memory_id, "works remotely", reason="told us her PIN is 482915")
    engine.delete(event.memory_id, reason="card 4111 1111 1111 1111 was cancelled")
    history = repr(engine.history(event.memory_id))
    assert "482915" not in history and "4111" not in history
    assert "told us her PIN is [redacted]" in history


def test_an_explicit_secret_is_masked_under_mask(store: PgStore) -> None:
    engine = _engine(store, "mask")
    engine.remember(Fact("wifi password is tiger-77"), ADA)
    assert _current(engine) == ["wifi password is [redacted]"]
