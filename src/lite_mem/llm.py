"""The LLM layer (design §3): every call names a role, configuration maps roles to a provider and
a model, and every answer is structured (a Pydantic model) and carries its provenance. No fallback
chain: a failed call raises `ModelError`.

A call sends a system prompt (a template from `prompts/`) and a payload; the client serialises the
payload (JSON), so callers and fakes cannot disagree about the user turn's format."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from lite_mem.config import LLMConfig, ModelTarget, Role
from lite_mem.errors import ConfigError, ModelError


@dataclass(frozen=True)
class Provenance:
    """Who answered: recorded with every change the answer produced."""

    role: Role
    provider: str
    model: str

    @property
    def tag(self) -> str:
        """`provider:model`, as stored in the history."""
        return f"{self.provider}:{self.model}"


class LLM(Protocol):
    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]: ...


def _failed(role: Role, target: ModelTarget, exc: BaseException) -> ModelError:
    """A call that failed, without the upstream message: SDKs and proxies can echo keys,
    authorization headers or credentialed URLs. The cause stays chained for debugging."""
    return ModelError(f"{role} ({target.provider}:{target.model}) failed: {type(exc).__name__}")


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """`model`'s JSON schema as strict structured output wants it: every property required (an
    optional one is still nullable), no extra properties, in every nested object too."""
    schema = model.model_json_schema()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
                for prop in node["properties"].values():
                    prop.pop("default", None)  # strict mode rejects defaults
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


def _parsed[T: BaseModel](role: Role, schema: type[T], raw: Any, provenance: Provenance) -> T:
    """The reply as the role's model, or ModelError (never a half-parsed answer)."""
    if raw is None:
        raise ModelError(f"{role}: no answer in the reply ({provenance.tag})")
    try:
        if isinstance(raw, str):
            return schema.model_validate_json(raw)
        return schema.model_validate(raw)
    except ValidationError as exc:
        raise ModelError(f"{role}: reply does not match {schema.__name__}") from exc


def _key(target: ModelTarget) -> str | None:
    return target.api_key.get_secret_value() if target.api_key else None


class _VendorLLM:
    """What both vendors' clients share: the config, one SDK client per (endpoint, key), and the
    user turn (the payload as JSON)."""

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self._config = config
        self._client = client  # tests: one fake client for every role
        self._clients: dict[tuple[str | None, str | None], Any] = {}

    def _sdk(self, target: ModelTarget) -> Any:
        if self._client is not None:
            return self._client
        slot = (target.base_url, _key(target))
        if slot not in self._clients:
            self._clients[slot] = self._make(target)
        return self._clients[slot]

    def _make(self, target: ModelTarget) -> Any:
        raise NotImplementedError

    @staticmethod
    def _user_turn(payload: Mapping[str, Any]) -> dict[str, str]:
        return {"role": "user", "content": json.dumps(dict(payload), ensure_ascii=False)}


class OpenAICompatibleLLM(_VendorLLM):
    """Chat completions with a strict JSON schema (`response_format`): any OpenAI-compatible
    endpoint — Ollama `/v1`, llm-gateway, a vendor."""

    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]:
        target = self._config.target(role)
        try:
            response = self._sdk(target).chat.completions.create(
                model=target.model,
                messages=[{"role": "system", "content": system}, self._user_turn(payload)],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": strict_schema(schema),
                        "strict": True,
                    },
                },
                timeout=target.timeout_s,
                # not a typed SDK argument for every model: sent as is when configured
                extra_body={"reasoning_effort": target.reasoning} if target.reasoning else None,
            )
        except Exception as exc:  # the SDK's errors, network errors: no fallback (rule 6)
            raise _failed(role, target, exc) from exc
        provenance = Provenance(role, target.provider, response.model or target.model)
        choice = response.choices[0] if response.choices else None
        if choice is not None and choice.finish_reason == "length":
            raise ModelError(f"{role}: reply truncated ({provenance.tag})")
        content = choice.message.content if choice is not None else None
        return _parsed(role, schema, content, provenance), provenance

    def _make(self, target: ModelTarget) -> Any:
        from openai import OpenAI

        return OpenAI(base_url=target.base_url, api_key=_key(target) or "unused")


class AnthropicLLM(_VendorLLM):
    """Anthropic Messages with one forced tool whose input schema is the role's output model (the
    `anthropic` extra)."""

    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]:
        target = self._config.target(role)
        name = schema.__name__
        try:
            response = self._sdk(target).messages.create(
                model=target.model,
                max_tokens=2048,
                system=system,
                messages=[self._user_turn(payload)],
                tools=[{"name": name, "input_schema": strict_schema(schema)}],
                tool_choice={"type": "tool", "name": name},
                timeout=target.timeout_s,
            )
        except Exception as exc:
            raise _failed(role, target, exc) from exc
        provenance = Provenance(role, target.provider, response.model or target.model)
        if response.stop_reason == "max_tokens":
            raise ModelError(f"{role}: reply truncated ({provenance.tag})")
        used = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        return _parsed(role, schema, used[0].input if used else None, provenance), provenance

    def _make(self, target: ModelTarget) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ConfigError("provider anthropic needs the extra: lite-mem[anthropic]") from exc
        return Anthropic(api_key=_key(target), base_url=target.base_url)


class RoutedLLM:
    """The LLM per role: each role goes to the client of its own provider (config.target)."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._by_provider: dict[str, LLM] = {}

    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]:
        provider = self._config.target(role).provider
        if provider not in self._by_provider:
            self._by_provider[provider] = (
                AnthropicLLM(self._config) if provider == "anthropic"
                else OpenAICompatibleLLM(self._config)
            )  # fmt: skip
        return self._by_provider[provider].structured(role, system, payload, schema)
