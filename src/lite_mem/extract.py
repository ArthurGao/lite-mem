"""Fact extraction (design §2, §3): the `extract` role proposes facts from a conversation; code
cleans them (AGENTS.md rule 5). A caller's own facts go through the engine's strict gate instead;
a model's are normalised here — clipped, de-duplicated and capped, never failing the batch."""

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from lite_mem import prompts
from lite_mem.llm import LLM
from lite_mem.types import Fact, Message

MAX_FACTS = 20  # per conversation: more is the model listing the chat, not memory
MAX_EXTRACTED_CHARS = 1000  # tighter than the engine's gate for a caller's fact (2000)
_CJK = re.compile(r"[\u3400-\u9fff]")

ExtractedType = Literal["semantic", "episodic", "procedural"]


class ExtractedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    type: ExtractedType
    importance: float


class ExtractedFacts(BaseModel):
    """The role's structured output (every field required: strict JSON schema)."""

    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedFact]


@dataclass(frozen=True)
class Extraction:
    facts: list[Fact]
    produced_by: str | None  # provider:model; None when no model was asked


class Extractor(Protocol):
    def extract(self, messages: list[Message]) -> Extraction: ...


class LlmExtractor:
    def __init__(self, llm: LLM, prompt: str = prompts.EXTRACT) -> None:
        self._llm = llm
        self._prompt = prompts.load(prompt)

    def extract(self, messages: list[Message]) -> Extraction:
        said = [{"role": m.role, "content": m.content} for m in messages if m.content.strip()]
        if not any(m["role"] == "user" for m in said):
            return Extraction([], None)  # nothing the person said: no call
        # Code, not the model, says which language the facts are written in: a small model
        # follows the (English) prompt's language otherwise (seen with qwen3:8b).
        spoken = " ".join(m["content"] for m in said if m["role"] == "user")
        language = "Chinese" if _CJK.search(spoken) else "the user's language"
        answer, provenance = self._llm.structured(
            "extract", self._prompt, {"LANGUAGE": language, "MESSAGES": said}, ExtractedFacts
        )
        return Extraction(normalise(answer.facts), provenance.tag)


def normalise(drafts: list[ExtractedFact]) -> list[Fact]:
    """The facts worth writing: trimmed and clipped, scores clamped to 0..1, blanks and repeats
    dropped (first kept), at most MAX_FACTS, in the model's order."""
    facts: dict[str, Fact] = {}
    for d in drafts:
        content = d.content.strip()[:MAX_EXTRACTED_CHARS].strip()
        if not content or content in facts:
            continue
        importance = min(max(d.importance, 0.0), 1.0)
        facts[content] = Fact(content, type=d.type, importance=importance)
        if len(facts) == MAX_FACTS:
            break
    return list(facts.values())
