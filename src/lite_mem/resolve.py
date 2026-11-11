"""Conflict resolution (design §2): for each new fact, the `resolve` role compares it with the
related memories of exactly the same scope and picks ADD, UPDATE (it replaces one), DELETE (one is
no longer true) or NOOP (nothing new). The model proposes, code decides: a target outside the
candidates is never applied — the fact is simply added."""

import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from lite_mem import prompts
from lite_mem.llm import LLM
from lite_mem.types import Memory

Action = Literal["ADD", "UPDATE", "DELETE", "NOOP"]
PROMPT = "resolve_conflict.j2"


class Decision(BaseModel):
    """The role's structured output. `target_id` names an EXISTING memory for UPDATE / DELETE."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    target_id: str | None
    reason: str


@dataclass(frozen=True)
class Resolution:
    action: Action
    target: uuid.UUID | None
    reason: str
    produced_by: str | None  # provider:model; None when code decided without a call


class Resolver(Protocol):
    def resolve(self, new: str, related: list[Memory]) -> Resolution: ...


class LlmResolver:
    def __init__(self, llm: LLM, prompt: str = PROMPT) -> None:
        self._llm = llm
        self._prompt = prompts.load(prompt)

    def resolve(self, new: str, related: list[Memory]) -> Resolution:
        if not related:
            return Resolution("ADD", None, "no related memory in this scope", None)
        payload = {
            "NEW": new,
            "EXISTING": [{"id": str(m.id), "fact": m.content} for m in related],
        }
        decision, provenance = self._llm.structured("resolve", self._prompt, payload, Decision)
        if decision.action in ("ADD", "NOOP"):
            return Resolution(decision.action, None, decision.reason, provenance.tag)
        try:
            target = uuid.UUID(decision.target_id or "")
        except ValueError:
            reason = f"model chose {decision.action} of {decision.target_id!r}, not an id; added"
            return Resolution("ADD", None, reason, provenance.tag)
        # Whether the target is among the candidates and still current: the engine decides.
        return Resolution(decision.action, target, decision.reason, provenance.tag)


class AddOnlyResolver:
    """Never resolves: every fact is added (the behaviour of mem0 2.2.1, for comparison)."""

    def resolve(self, new: str, related: list[Memory]) -> Resolution:
        return Resolution("ADD", None, "add-only resolver", None)
