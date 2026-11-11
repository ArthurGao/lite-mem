"""Consolidation (design §6): the `consolidate` role judges whether similar memories say one
thing and words the merged fact, and summarises old episodes into durable facts. The model
proposes; the engine decides what is written (which memories are candidates, that sources are
among them, each used once, and that every new fact passes the same screening as any other)."""

import uuid
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from lite_mem import prompts
from lite_mem.llm import LLM
from lite_mem.types import Memory


class _MergeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    same: bool
    content: str | None
    reason: str


class _SummaryFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    sources: list[str]


class _SummaryAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[_SummaryFact]


@dataclass(frozen=True)
class MergeDecision:
    content: str | None  # None: they are different facts, nothing is merged
    reason: str
    produced_by: str | None


@dataclass(frozen=True)
class Summary:
    content: str
    sources: tuple[uuid.UUID, ...]  # ids among the episodes given, each used once
    produced_by: str | None


class Consolidator(Protocol):
    def merge(self, memories: list[Memory]) -> MergeDecision: ...

    def summarise(self, episodes: list[Memory]) -> list[Summary]: ...


class LlmConsolidator:
    def __init__(
        self,
        llm: LLM,
        merge_prompt: str = "consolidate_merge.j2",
        summarise_prompt: str = "consolidate_summarise.j2",
    ) -> None:
        self._llm = llm
        self._merge = prompts.load(merge_prompt)
        self._summarise = prompts.load(summarise_prompt)

    def merge(self, memories: list[Memory]) -> MergeDecision:
        payload = {"FACTS": prompts.listed(memories)}
        answer, provenance = self._llm.structured("consolidate", self._merge, payload, _MergeAnswer)
        content = (answer.content or "").strip() if answer.same else ""
        return MergeDecision(content or None, answer.reason, provenance.tag)

    def summarise(self, episodes: list[Memory]) -> list[Summary]:
        payload = {"EPISODES": prompts.listed(episodes)}
        answer, provenance = self._llm.structured(
            "consolidate", self._summarise, payload, _SummaryAnswer
        )
        given = {str(m.id): m.id for m in episodes}
        used: set[uuid.UUID] = set()
        summaries: list[Summary] = []
        for fact in answer.facts:  # code decides: only given ids, each once, and some content
            sources = tuple(
                dict.fromkeys(given[s] for s in fact.sources if s in given and given[s] not in used)
            )
            if fact.content.strip() and sources:
                used.update(sources)
                summaries.append(Summary(fact.content.strip(), sources, provenance.tag))
        return summaries
