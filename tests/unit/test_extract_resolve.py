import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx2
import lite_model
import pytest

from lite_mem.config import LLMConfig
from lite_mem.errors import ConfigError, ModelError
from lite_mem.extract import (
    MAX_FACTS,
    MAX_LEFT_OUT,
    ExtractedFact,
    LlmExtractor,
    mentioned,
    normalise,
)
from lite_mem.llm import LiteModelLLM
from lite_mem.resolve import Decision, LlmResolver
from lite_mem.testing import FakeLLM, RuleExtractor, memory, user_lines
from lite_mem.types import Message
from tests.conftest import openai_reply

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_normalise_clips_dedupes_and_caps() -> None:
    drafts = [
        ExtractedFact(content="  Lives in Berlin ", type="semantic", importance=1.7),
        ExtractedFact(content="Lives in Berlin", type="episodic", importance=0.1),
        ExtractedFact(content="   ", type="semantic", importance=0.5),
        ExtractedFact(content="x" * 1200, type="procedural", importance=-1),
    ]
    facts = normalise(drafts)
    assert [(f.content[:15], f.type, f.importance) for f in facts] == [
        ("Lives in Berlin", "semantic", 1.0),
        ("x" * 15, "procedural", 0.0),
    ]
    assert len(facts[1].content) == 1000
    many = [ExtractedFact(content=f"f{i}", type="semantic", importance=0.5) for i in range(30)]
    assert len(normalise(many)) == MAX_FACTS


def test_the_extractor_sends_the_conversation_and_tags_the_facts() -> None:
    llm = FakeLLM(extract=user_lines)
    out = LlmExtractor(llm).extract([Message("user", "我住北京"), Message("assistant", "好的")])
    assert [f.content for f in out.facts] == ["我住北京"] and out.produced_by == "fake:fake-model"
    [(role, payload)] = llm.calls
    assert role == "extract" and payload["MESSAGES"][1] == {"role": "assistant", "content": "好的"}
    assert payload["LANGUAGE"] == "Chinese"  # decided by code from what the user wrote


def test_no_user_turn_means_no_call() -> None:
    llm = FakeLLM()
    assert LlmExtractor(llm).extract([Message("assistant", "hi")]).facts == []
    assert llm.calls == []
    assert [f.content for f in RuleExtractor().extract([Message("user", "a")]).facts] == ["a"]


def test_the_resolver_decides_in_code() -> None:
    old = memory("Lives in Berlin")
    nothing = FakeLLM()
    assert LlmResolver(nothing).resolve("x", []).action == "ADD" and nothing.calls == []

    def answer(action: str, target: str | None) -> FakeLLM:
        return FakeLLM(resolve=lambda p: {"action": action, "target_id": target, "reason": "r"})

    moved = LlmResolver(answer("UPDATE", str(old.id))).resolve("Lives in Paris", [old])
    assert (moved.action, moved.target, moved.produced_by) == ("UPDATE", old.id, "fake:fake-model")
    garbled = LlmResolver(answer("DELETE", "the cat one")).resolve("x", [old])
    assert (garbled.action, garbled.target) == ("ADD", None) and "not an id" in garbled.reason
    # an id outside the candidates is passed on: the engine refuses it (test_infer.py)
    stray = uuid.uuid4()
    assert LlmResolver(answer("DELETE", str(stray))).resolve("x", [old]).target == stray
    same = LlmResolver(answer("NOOP", None)).resolve("Lives in Berlin", [old])
    assert (same.action, same.target) == ("NOOP", None)


def test_an_update_can_keep_the_part_of_the_old_fact_that_still_holds() -> None:
    old = memory("lives in Wellington and works as a nurse")
    answer = {"action": "UPDATE", "target_id": str(old.id), "reason": "moved",
              "remainder": "works as a nurse"}  # fmt: skip
    llm = FakeLLM(resolve=lambda p: answer)
    resolution = LlmResolver(llm).resolve("moved to Auckland", [old])
    assert (resolution.action, resolution.remainder) == ("UPDATE", "works as a nurse")


# The adapter onto lite-model (design §3). Requests go to a recording transport, never a network.
_DECISION = {"action": "NOOP", "target_id": None, "reason": "r", "remainder": None}


def _built(data: dict[str, Any], handler: Any) -> LiteModelLLM:
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LiteModelLLM.build(LLMConfig.model_validate({"max_retries": 0, **data}), http)


def test_each_role_goes_to_its_own_backend_and_records_who_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITEMEM_TEST_KEY", "k")
    hosts: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        hosts.append(request.url.host)
        return openai_reply(f"{request.url.host}-model", json.dumps(_DECISION))

    llm = _built({
        "roles": {"resolve": {"backend": "vendor", "model": "big"}},
        "backends": {"vendor": {"type": "openai", "base_url": "http://vendor.test/v1",
                                "api_key_env": "LITEMEM_TEST_KEY"}},
    }, handler)  # fmt: skip
    _, extracted = llm.structured("extract", "s", {}, Decision)
    _, resolved = llm.structured("resolve", "s", {}, Decision)
    assert hosts == ["127.0.0.1", "vendor.test"]
    # the model the response names, not the configured one
    assert (extracted.tag, resolved.tag) == ("openai:127.0.0.1-model", "vendor:vendor.test-model")


def test_a_bad_config_is_refused_before_any_call() -> None:
    with pytest.raises(ConfigError, match="unknown backend"):
        LiteModelLLM.build(LLMConfig.model_validate({"roles": {"resolve": {"backend": "nope"}}}))


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (openai_reply("m", "{}", finish="length"), lite_model.Truncated),
        (openai_reply("m", "not json"), lite_model.BadStructuredReply),
        (httpx2.Response(500, json={"error": "boom"}), lite_model.CallFailed),
    ],
)
def test_an_unusable_reply_is_lite_mems_model_error(
    reply: httpx2.Response, error: type[Exception]
) -> None:
    llm = _built({}, lambda request: reply)
    with pytest.raises(ModelError) as caught:
        llm.structured("resolve", "s", {}, Decision)
    assert type(caught.value) is ModelError and isinstance(caught.value.__cause__, error)


def test_a_provider_error_never_carries_its_text_out() -> None:
    leaky = "401 for https://u:pw@proxy/v1 with Authorization: Bearer sk-secret-123"

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(leaky, request=request)

    with pytest.raises(ModelError) as caught:
        _built({}, handler).structured("resolve", "s", {}, Decision)
    assert "sk-secret" not in str(caught.value) and "pw@" not in str(caught.value)
    assert "resolve" in str(caught.value) and caught.value.__cause__ is not None  # kept inside


def test_no_role_may_use_a_routing_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    # a memory written by a model a router picked could not be compared with the configured one's
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    routed = {"extract": {"backend": "litellm", "model": "smart-router"}}
    config = LLMConfig.model_validate({"roles": routed})
    with pytest.raises(ConfigError, match="routing alias and the role is not routable"):
        LiteModelLLM.build(config)


@pytest.mark.parametrize(
    ("fact", "compound"),
    [
        ("Lives in Wellington and works as a nurse", True),
        ("Works at Google as a software engineer, lives in Sydney", True),
        ("住在北京，在银行做会计", True),
        ("喜欢绿茶和咖啡", True),
        ("Works part-time at a cafe as a barista", False),
        ("Has a dog named Max", False),
        ("Works at a brand agency in Auckland", False),  # "and" inside a word
        ("在银行做会计", False),
        ("住在和平路", False),  # 和 inside a word, not joining two statements
        ("性格温和", False),
    ],
)
def test_only_a_compound_fact_can_leave_a_remainder(fact: str, compound: bool) -> None:
    from lite_mem.engine import _compound

    assert _compound(fact) is compound


@pytest.mark.parametrize(
    ("said", "names"),
    [
        ("My wife Anna and I have a dog named Max.", ["Anna", "Max"]),
        ("Hi, I'm Emma. I live in Auckland and work as a nurse.", ["Emma", "Auckland"]),
        ("Lucy moved to Paris last month.", ["Paris"]),  # a sentence's first word is not judged
        ("Actually I quit the bank, I'm a freelance ML consultant now.", []),
        ("我和老公小李住在杭州，我们有个女儿叫朵朵。", ["杭州", "朵朵"]),
        ("你好，我叫小王，住在北京。", ["北京", "小王"]),
        (
            "We visited my aunt Élodie in Lyon; I've met her McDonald cousins.",
            ["Élodie", "Lyon", "McDonald"],
        ),
        ("Sadly I'M NOT SURE, said ANNA.", []),
        ("我最近喜欢上了绿茶，每天早上都喝一杯。", []),
    ],
)
def test_the_names_and_places_a_message_mentions(said: str, names: list[str]) -> None:
    from lite_mem.extract import mentioned

    assert mentioned(said) == names


def _drops(name: str) -> Any:
    """An `extract` handler that leaves out the fact naming `name` on the first try only, and lists
    every other user line as a fact."""

    def answer(payload: dict[str, Any]) -> dict[str, Any]:
        said = [m["content"] for m in payload["MESSAGES"] if m["role"] == "user"]
        facts = [f for s in said for f in s.split(" / ")]
        if "LEFT_OUT" not in payload:
            facts = [f for f in facts if name not in f]
        return {"facts": [{"content": f, "type": "semantic", "importance": 0.5} for f in facts]}

    return answer


def test_a_name_the_answer_left_out_is_asked_for_once_more() -> None:
    """qwen3:8b, 2026-11-20: "My wife Anna and I have a dog named Max" kept only the dog (7 of 8
    runs). Code finds the names and places the person said; one the facts do not mention is
    asked for once more, and the facts about it join the first answer."""
    llm = FakeLLM(extract=_drops("Anna"))
    out = LlmExtractor(llm).extract([Message("user", "Wife is Anna / Has a dog named Max")])
    assert [f.content for f in out.facts] == ["Has a dog named Max", "Wife is Anna"]
    assert out.produced_by == "fake:fake-model"
    first, again = [p for _, p in llm.calls]
    assert "LEFT_OUT" not in first and again["LEFT_OUT"] == ["Anna"]
    assert again["MESSAGES"] == first["MESSAGES"]
    systems = [c.messages[0].content for c in llm.fake.calls]
    assert systems[1].startswith(systems[0]) and "did not mention" in systems[1]


def test_an_answer_that_covers_every_name_is_not_asked_again() -> None:
    llm = FakeLLM(extract=_drops("nobody"))
    LlmExtractor(llm).extract([Message("user", "Wife is Anna / Has a dog named Max")])
    assert len(llm.calls) == 1


def test_the_second_answer_only_adds_facts_about_the_names_left_out() -> None:
    """The first answer is never replaced: a retry that rewords or drops its facts changes
    nothing but the missing name's facts."""

    def retry(payload: dict[str, Any]) -> dict[str, Any]:
        if "LEFT_OUT" not in payload:
            return _drops("Anna")(payload)
        said = ["Owns a dog called Max", "Wife is Anna", "Loves jazz"]
        return {"facts": [{"content": c, "type": "semantic", "importance": 0.5} for c in said]}

    llm = FakeLLM(extract=retry)
    out = LlmExtractor(llm).extract([Message("user", "Wife is Anna / Has a dog named Max")])
    assert [f.content for f in out.facts] == ["Has a dog named Max", "Wife is Anna"]
    assert len(llm.calls) == 2  # asked once more, never a third time


def test_the_recheck_can_be_turned_off() -> None:
    llm = FakeLLM(extract=_drops("Anna"))
    out = LlmExtractor(llm, left_out_prompt=None).extract(
        [Message("user", "Wife is Anna / Has a dog named Max")]
    )
    assert [f.content for f in out.facts] == ["Has a dog named Max"] and len(llm.calls) == 1


def test_the_names_asked_for_again_are_capped() -> None:
    """A message listing many names sends at most MAX_LEFT_OUT of them back, in the order said."""
    names = [f"Person{chr(97 + i)}" for i in range(MAX_LEFT_OUT + 5)]
    llm = FakeLLM(extract=lambda _payload: {"facts": []})
    LlmExtractor(llm).extract([Message("user", " / ".join(f"knows {n}" for n in names))])
    assert llm.calls[1][1]["LEFT_OUT"] == names[:MAX_LEFT_OUT]


def test_finding_names_is_linear_in_the_text() -> None:
    """Up to MAX_MESSAGES x MAX_MESSAGE_CHARS of user text reaches `mentioned`: judging each word
    by copying the text before it was quadratic (security review, 2026-11-20)."""
    text = "a Bob. " * 150_000  # ~1M characters, 150k capitalised words
    started = time.perf_counter()
    assert mentioned(text) == ["Bob"]
    assert time.perf_counter() - started < 2
