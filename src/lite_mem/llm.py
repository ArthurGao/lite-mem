"""The LLM layer (design §3): every call names a role and gets a structured answer (a Pydantic
model) with its provenance. The calls themselves are lite-model's (its design §12): it maps the
role to a model, sends the request, retries on the same model only (no fallback chain) and reads
which model answered. This module keeps lite-mem's own `LLM` Protocol (rule 1) and the adapter.

A call sends a system prompt (a template from `prompts/`) and a payload; the adapter serialises
the payload (JSON), so callers and fakes cannot disagree about the user turn's format."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, get_args

import lite_model
from lite_model import Message, RolePolicy, RoleSpec
from pydantic import BaseModel

from lite_mem.config import LLMConfig, Role
from lite_mem.errors import ConfigError, ModelError

if TYPE_CHECKING:
    import httpx2

# No role may use a routing alias (lite-model design §7): a memory written by a model the router
# picked could not be compared with one written by the configured model.
ROLES = RoleSpec({role: RolePolicy(routable=False) for role in get_args(Role)})


@dataclass(frozen=True)
class Provenance:
    """Who answered: recorded with every change the answer produced."""

    role: Role
    # `provider:model`, as stored in the history: lite-model's backend name ("openai" by default)
    # and the model that answered, read from the response (`model?` when the endpoint did not say)
    tag: str


class LLM(Protocol):
    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]: ...


def build_model(
    config: lite_model.ModelsConfig,
    http_client: "httpx2.Client | None" = None,
    roles: RoleSpec = ROLES,
) -> lite_model.LiteModel:
    """lite-model over lite-mem's roles (the LLM's, or the embedder's). The whole config is
    checked before any call (an unknown backend, a missing key or extra, a routing alias) and
    refused as lite-mem's ConfigError."""
    try:
        return lite_model.build(config, roles, http_client=http_client)
    except lite_model.ConfigError as exc:
        raise ConfigError(str(exc)) from exc


class LiteModelLLM:
    """lite-mem's `LLM` over a `lite_model.Model`. lite-model's errors come out as lite-mem's own
    (their messages never carry upstream text; the cause stays chained)."""

    def __init__(self, model: lite_model.Model) -> None:
        self._model = model

    @classmethod
    def build(cls, config: LLMConfig, http_client: "httpx2.Client | None" = None) -> "LiteModelLLM":
        return cls(build_model(config, http_client))

    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]:
        user = json.dumps(dict(payload), ensure_ascii=False)
        try:
            reply = self._model.structured(
                role, [Message("system", system), Message("user", user)], schema
            )
        except lite_model.ModelError as exc:
            raise ModelError(str(exc)) from exc
        return reply.value, Provenance(role, reply.provenance.tag)
