"""Fact extraction (design §2, §3): the `extract` role proposes facts from a conversation; code
cleans them (AGENTS.md rule 5). A caller's own facts go through the engine's strict gate instead;
a model's are normalised here — clipped, de-duplicated and capped, never failing the batch."""

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from lite_mem import prompts
from lite_mem.keyword import places
from lite_mem.llm import LLM
from lite_mem.types import Fact, Message

MAX_FACTS = 20  # per conversation: more is the model listing the chat, not memory
MAX_EXTRACTED_CHARS = 1000  # tighter than the engine's gate for a caller's fact (2000)
_HAN = "\u3400-\u9fff"
_CJK = re.compile(f"[{_HAN}]")
# Names and places a message mentions (`mentioned`): a capitalised word that does not open a
# sentence; a place jieba tags `ns`; a Chinese name after 叫 ("女儿叫朵朵"). jieba's `nr` (a
# person) is not used: it tags kinship words (老公) and splits names (小李 -> 李).
_WORD = re.compile(r"[^\W\d_][\w'-]*")
_CALLED = re.compile(rf"(?:名叫|叫做|叫)([{_HAN}]{{2,3}}?)(?=[，。,.！!？?、；;\s]|$)")
_SENTENCE_ENDS = ".!?。！？"

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
    def __init__(
        self,
        llm: LLM,
        prompt: str = prompts.EXTRACT,
        left_out_prompt: str | None = prompts.EXTRACT_LEFT_OUT,
    ) -> None:
        self._llm = llm
        self._prompt = prompts.load(prompt)
        self._left_out_prompt = prompts.load(left_out_prompt) if left_out_prompt else None

    def extract(self, messages: list[Message]) -> Extraction:
        said = [{"role": m.role, "content": m.content} for m in messages if m.content.strip()]
        if not any(m["role"] == "user" for m in said):
            return Extraction([], None)  # nothing the person said: no call
        # Code, not the model, says which language the facts are written in: a small model
        # follows the (English) prompt's language otherwise (seen with qwen3:8b).
        spoken = " ".join(m["content"] for m in said if m["role"] == "user")
        language = "Chinese" if _CJK.search(spoken) else "the user's language"
        payload = {"LANGUAGE": language, "MESSAGES": said}
        answer, provenance = self._llm.structured("extract", self._prompt, payload, ExtractedFacts)
        facts = answer.facts
        # The model proposes, code checks: a name or place the person said that no fact mentions
        # is asked for once more (qwen3:8b kept "has a dog" of "my wife Anna and I have a dog").
        # The first answer stays; only the new facts about those names join it.
        recheck = self._left_out_prompt
        missing = _left_out(mentioned(spoken), facts) if recheck is not None else []
        if recheck is not None and missing:
            again, provenance = self._llm.structured(  # same role and model: one tag for both
                "extract", self._prompt + recheck, {**payload, "LEFT_OUT": missing}, ExtractedFacts
            )
            facts = facts + [f for f in again.facts if len(_left_out(missing, [f])) < len(missing)]
        return Extraction(normalise(facts), provenance.tag)


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


def mentioned(text: str) -> list[str]:
    """The names and places `text` mentions, each once (see `_WORD`). A sentence's first word is
    not judged ("Actually", "Lucy"): missing a name costs nothing, a wrong one costs a call."""
    latin = [
        m.group().removesuffix("'s")
        for m in _WORD.finditer(text)
        if _proper(m.group()) and not _opens_sentence(text, m.start())
    ]
    chinese = [*places(text), *(m.group(1) for m in _CALLED.finditer(text))]
    return list(dict.fromkeys([*latin, *chinese] if _CJK.search(text) else latin))


def _proper(word: str) -> bool:
    """Capitalised like a name: "Anna", "Élodie", "McDonald"; not "I'm", "ANNA" or 小王."""
    return len(word) > 1 and word[0].isupper() and word[1].islower()


def _opens_sentence(text: str, at: int) -> bool:
    before = text[:at].rstrip()
    return not before or before[-1] in _SENTENCE_ENDS


def _left_out(names: list[str], facts: list[ExtractedFact]) -> list[str]:
    said = " ".join(f.content for f in facts).lower()
    return [n for n in names if n.lower() not in said]
