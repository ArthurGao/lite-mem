"""The request bodies lite-mem sends for each role (lite-model design §12.2 step 1).

`tests/data/llm_requests.json` was captured from the client lite-mem had before it moved onto
lite-model (lite-mem fce155b); the client it has now must send the same JSON. Only `_llm` knows
how a client is built."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel

from lite_mem import prompts
from lite_mem.config import LLMConfig
from lite_mem.consolidate import _MergeAnswer, _SummaryAnswer
from lite_mem.extract import ExtractedFacts
from lite_mem.llm import LLM, LiteModelLLM
from lite_mem.resolve import Decision
from tests.conftest import openai_reply

RECORDED = Path(__file__).parents[1] / "data" / "llm_requests.json"

# name -> (role, system prompt template, payload, schema, a valid answer)
CASES: dict[str, tuple[str, str, dict[str, Any], type[BaseModel], dict[str, Any]]] = {
    "extract": (
        "extract", prompts.EXTRACT,
        {"LANGUAGE": "Chinese", "MESSAGES": [{"role": "user", "content": "我上个月搬到上海了"}]},
        ExtractedFacts, {"facts": []},
    ),
    "resolve": (
        "resolve", prompts.RESOLVE,
        {"NEW": "Lives in Auckland", "EXISTING": "- [7] Lives in Wellington"},
        Decision, {"action": "NOOP", "target_id": None, "reason": "r", "remainder": None},
    ),
    "merge": (
        "consolidate", prompts.MERGE,
        {"FACTS": "- [1] 每天喝绿茶\n- [2] 每天早上喝一杯绿茶"},
        _MergeAnswer, {"same": False, "content": None, "reason": "r"},
    ),
    "summarise": (
        "consolidate", prompts.SUMMARISE,
        {"EPISODES": "- [1] went to the gym on Monday"},
        _SummaryAnswer, {"facts": []},
    ),
}  # fmt: skip

# The two OpenAI-compatible configurations pinned: the defaults (local Ollama /v1, qwen3:8b), and
# reasoning set globally with a role overriding it. Anthropic: one model for every role.
OPENAI_CONFIGS = ("default", "reasoning")

Send = Callable[[httpx2.Request], httpx2.Response]


def _openai_reply(request: httpx2.Request) -> httpx2.Response:
    answer = _answer_for(json.loads(request.content)["response_format"]["json_schema"]["name"])
    return openai_reply("qwen3:8b", json.dumps(answer))


def _anthropic_reply(request: httpx2.Request) -> httpx2.Response:
    name = json.loads(request.content)["tool_choice"]["name"]
    return httpx2.Response(200, json={
        "id": "m", "type": "message", "role": "assistant", "model": "claude-x",
        "stop_reason": "tool_use", "stop_sequence": None,
        "content": [{"type": "tool_use", "id": "t", "name": name, "input": _answer_for(name)}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })  # fmt: skip


def _answer_for(schema_name: str) -> dict[str, Any]:
    return next(c[4] for c in CASES.values() if c[3].__name__ == schema_name)


def _llm(config: str, http: httpx2.Client, monkeypatch: pytest.MonkeyPatch) -> LLM:
    """The client under test, built the way lite-mem builds it, over a recording transport."""
    if config == "anthropic":
        monkeypatch.setenv("LITEMEM_TEST_ANTHROPIC_KEY", "test-key")
        data: dict[str, Any] = {
            "backend": "claude", "model": "claude-x", "max_output_tokens": 2048,
            "backends": {"claude": {"type": "anthropic", "base_url": "http://anthropic.test",
                                    "api_key_env": "LITEMEM_TEST_ANTHROPIC_KEY"}},
        }  # fmt: skip
    elif config == "reasoning":
        data = {"reasoning_effort": "none", "roles": {"resolve": {"reasoning_effort": "high"}}}
    else:
        data = {}
    return LiteModelLLM.build(LLMConfig.model_validate(data), http_client=http)


def record(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Every case through every configuration: what reached the wire."""
    seen: dict[str, Any] = {}
    for config in (*OPENAI_CONFIGS, "anthropic"):
        sent: list[httpx2.Request] = []
        reply: Send = _anthropic_reply if config == "anthropic" else _openai_reply

        def handler(
            request: httpx2.Request, reply: Send = reply, sent: list[httpx2.Request] = sent
        ) -> httpx2.Response:
            sent.append(request)
            return reply(request)

        llm = _llm(config, httpx2.Client(transport=httpx2.MockTransport(handler)), monkeypatch)
        for name, (role, template, payload, schema, answer) in CASES.items():
            value, _ = llm.structured(role, prompts.load(template), payload, schema)
            assert value.model_dump() == answer
            request = sent[-1]
            seen[f"{config}/{name}"] = {
                "method": request.method,
                "url": str(request.url),
                "auth": request.headers.get("authorization") or request.headers.get("x-api-key"),
                "body": json.loads(request.content),
            }
        assert len(sent) == len(CASES)  # one request per call: no retry, no second call
    return seen


def test_every_role_sends_the_recorded_request(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    seen = record(monkeypatch)
    assert sorted(seen) == sorted(recorded)
    for key in recorded:
        assert seen[key] == recorded[key], key
