"""The LLM layer (design §3): every call names a role, configuration maps roles to a provider and
a model, and every answer is structured (a Pydantic model) and carries its provenance. No fallback
chain: a failed call raises `ModelError`.

A call sends a system prompt (a template from `prompts/`) and a payload; the client serialises the
payload (JSON), so callers and fakes cannot disagree about the user turn's format."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from lite_mem.config import Role


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
