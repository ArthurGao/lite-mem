import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from lite_mem.config import LLMConfig
from lite_mem.errors import ModelError
from lite_mem.extract import MAX_FACTS, ExtractedFact, LlmExtractor, RuleExtractor, normalise
from lite_mem.llm import AnthropicLLM, OpenAICompatibleLLM, RoutedLLM
from lite_mem.resolve import Decision, LlmResolver
from lite_mem.testing import FakeLLM, user_lines
from lite_mem.types import Memory, Message, Scope

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _memory(content: str) -> Memory:
    return Memory(
        id=uuid.uuid4(), scope=Scope("ada"), type="semantic", key=None, content=content,
        importance=0.5, confidence=0.8, valid_from=NOW, valid_to=None, superseded_by=None,
        expires_at=None, source=None, metadata={}, access_count=0, created_at=NOW, updated_at=NOW,
    )  # fmt: skip


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
    assert [f.content for f in out.facts] == ["我住北京"] and out.produced_by == "fake:fake-llm"
    [(role, payload)] = llm.calls
    assert role == "extract" and payload["MESSAGES"][1] == {"role": "assistant", "content": "好的"}
    assert payload["LANGUAGE"] == "Chinese"  # decided by code from what the user wrote


def test_no_user_turn_means_no_call() -> None:
    llm = FakeLLM()
    assert LlmExtractor(llm).extract([Message("assistant", "hi")]).facts == []
    assert llm.calls == []
    assert [f.content for f in RuleExtractor().extract([Message("user", "a")]).facts] == ["a"]


def test_the_resolver_decides_in_code() -> None:
    old = _memory("Lives in Berlin")
    nothing = FakeLLM()
    assert LlmResolver(nothing).resolve("x", []).action == "ADD" and nothing.calls == []

    def answer(action: str, target: str | None) -> FakeLLM:
        return FakeLLM(resolve=lambda p: {"action": action, "target_id": target, "reason": "r"})

    moved = LlmResolver(answer("UPDATE", str(old.id))).resolve("Lives in Paris", [old])
    assert (moved.action, moved.target, moved.produced_by) == ("UPDATE", old.id, "fake:fake-llm")
    garbled = LlmResolver(answer("DELETE", "the cat one")).resolve("x", [old])
    assert (garbled.action, garbled.target) == ("ADD", None) and "not an id" in garbled.reason
    # an id outside the candidates is passed on: the engine refuses it (test_infer.py)
    stray = uuid.uuid4()
    assert LlmResolver(answer("DELETE", str(stray))).resolve("x", [old]).target == stray
    same = LlmResolver(answer("NOOP", None)).resolve("Lives in Berlin", [old])
    assert (same.action, same.target) == ("NOOP", None)


class _Completions:
    def __init__(self, reply: Any) -> None:
        self.reply, self.sent = reply, []

    def create(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _openai(content: str, finish: str = "stop") -> tuple[Any, _Completions]:
    reply = SimpleNamespace(
        model="m-1",
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))],
    )
    completions = _Completions(reply)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_the_openai_compatible_client_sends_a_strict_schema() -> None:
    body = json.dumps({"action": "ADD", "target_id": None, "reason": "r"})
    client, completions = _openai(body)
    llm = OpenAICompatibleLLM(LLMConfig(model="m"), client)
    decision, provenance = llm.structured("resolve", "sys", {"NEW": "x"}, Decision)
    assert decision.action == "ADD" and provenance.tag == "openai:m-1"
    [sent] = completions.sent
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert json.loads(sent["messages"][1]["content"]) == {"NEW": "x"}
    schema = sent["response_format"]["json_schema"]["schema"]
    # strict mode: every property required (an optional one as nullable), nothing extra
    assert set(schema["required"]) == set(schema["properties"]) >= {"remainder"}
    assert schema["additionalProperties"] is False


def test_an_update_can_keep_the_part_of_the_old_fact_that_still_holds() -> None:
    memory = _memory("lives in Wellington and works as a nurse")
    body = json.dumps({"action": "UPDATE", "target_id": str(memory.id), "reason": "moved",
                       "remainder": "works as a nurse"})  # fmt: skip
    client, _ = _openai(body)
    resolver = LlmResolver(OpenAICompatibleLLM(LLMConfig(model="m"), client))
    resolution = resolver.resolve("moved to Auckland", [memory])
    assert (resolution.action, resolution.remainder) == ("UPDATE", "works as a nurse")


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (_openai("{}", finish="length")[0], "truncated"),
        (_openai("not json")[0], "does not match"),
        (
            SimpleNamespace(chat=SimpleNamespace(completions=_Completions(OSError("down")))),
            "failed",
        ),
    ],
)
def test_an_unusable_reply_raises(client: Any, message: str) -> None:
    llm = OpenAICompatibleLLM(LLMConfig(), client)
    with pytest.raises(ModelError, match=message):
        llm.structured("resolve", "sys", {}, Decision)


def test_the_anthropic_client_forces_the_tool_and_reads_its_input() -> None:
    block = SimpleNamespace(
        type="tool_use", input={"action": "NOOP", "target_id": None, "reason": "r"}
    )
    reply = SimpleNamespace(model="claude-x", stop_reason="tool_use", content=[block])
    messages = _Completions(reply)
    llm = AnthropicLLM(
        LLMConfig(provider="anthropic", model="claude-x"), SimpleNamespace(messages=messages)
    )
    decision, provenance = llm.structured("resolve", "sys", {}, Decision)
    assert decision.action == "NOOP" and provenance.tag == "anthropic:claude-x"
    assert messages.sent[0]["tool_choice"] == {"type": "tool", "name": "Decision"}


def test_routing_follows_each_roles_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMConfig.model_validate(
        {"roles": {"resolve": {"provider": "anthropic", "model": "claude-x"}}}
    )
    routed = RoutedLLM(config)
    seen: list[str] = []
    monkeypatch.setattr(OpenAICompatibleLLM, "structured", lambda *a, **k: seen.append("openai"))
    monkeypatch.setattr(AnthropicLLM, "structured", lambda *a, **k: seen.append("anthropic"))
    routed.structured("extract", "s", {}, Decision)
    routed.structured("resolve", "s", {}, Decision)
    assert seen == ["openai", "anthropic"]


def test_a_provider_error_never_carries_its_text_out() -> None:
    leaky = OSError("401 for https://u:pw@proxy/v1 with Authorization: Bearer sk-secret-123")
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(leaky)))
    with pytest.raises(ModelError) as caught:
        OpenAICompatibleLLM(LLMConfig(), client).structured("resolve", "s", {}, Decision)
    assert "sk-secret" not in str(caught.value) and "pw@" not in str(caught.value)
    assert "OSError" in str(caught.value) and caught.value.__cause__ is leaky  # kept inside


def test_an_empty_reply_is_a_model_error() -> None:
    empty = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(
        SimpleNamespace(model="m", choices=[])
    )))  # fmt: skip
    with pytest.raises(ModelError, match="no answer"):
        OpenAICompatibleLLM(LLMConfig(), empty).structured("resolve", "s", {}, Decision)
    no_tool = SimpleNamespace(messages=_Completions(
        SimpleNamespace(model="c", stop_reason="end_turn", content=[SimpleNamespace(type="text")])
    ))  # fmt: skip
    llm = AnthropicLLM(LLMConfig(provider="anthropic"), no_tool)
    with pytest.raises(ModelError, match="no answer"):
        llm.structured("resolve", "s", {}, Decision)


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    body = json.dumps({"action": "ADD", "target_id": None, "reason": "r"})
    client, completions = _openai(body)
    OpenAICompatibleLLM(LLMConfig(), client).structured("resolve", "s", {}, Decision)
    config = LLMConfig.model_validate(
        {"reasoning": "none", "roles": {"resolve": {"reasoning": "high"}}}
    )
    OpenAICompatibleLLM(config, client).structured("extract", "s", {}, Decision)
    OpenAICompatibleLLM(config, client).structured("resolve", "s", {}, Decision)
    cleared = LLMConfig.model_validate(
        {"reasoning": "none", "roles": {"resolve": {"reasoning": "default"}}}
    )  # a role opts back into the endpoint's own default
    OpenAICompatibleLLM(cleared, client).structured("resolve", "s", {}, Decision)
    assert [c["extra_body"] for c in completions.sent] == [
        None,
        {"reasoning_effort": "none"},
        {"reasoning_effort": "high"},
        None,
    ]
