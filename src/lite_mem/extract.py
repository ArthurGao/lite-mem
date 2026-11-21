"""Fact extraction (design §2, §3): the `extract` role proposes facts from a conversation; code
cleans them (AGENTS.md rule 5). A caller's own facts go through the engine's strict gate instead;
a model's are normalised here — clipped, de-duplicated and capped, never failing the batch."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict

from lite_mem import prompts
from lite_mem.keyword import places, terms
from lite_mem.llm import LLM
from lite_mem.sensitive import find
from lite_mem.types import Fact, Message

MAX_FACTS = 20  # per conversation: more is the model listing the chat, not memory
MAX_EXTRACTED_CHARS = 1000  # tighter than the engine's gate for a caller's fact (2000)
MAX_LEFT_OUT = 20  # names asked for once more: a long list is a roster, not what was dropped
_HAN = "\u3400-\u9fff"
_CJK = re.compile(f"[{_HAN}]")
# Names and places a message mentions (`mentioned`): a capitalised word that does not open a
# sentence; a place jieba tags `ns`; a Chinese name after 叫 ("女儿叫朵朵"). jieba's `nr` (a
# person) is not used: it tags kinship words (老公) and splits names (小李 -> 李).
_WORD = re.compile(r"[^\W\d_][\w'-]*")
_CALLED = re.compile(rf"(?:名叫|叫做|叫)([{_HAN}]{{2,3}}?)(?=[，。,.！!？?、；;\s]|$)")
_SENTENCE_ENDS = ".!?。！？"
# Statements a message makes (`statements`): split at clause punctuation (a comma or full stop
# only before a space or the end, so "1,000" and "3.5" stay whole) and at " and " / " but ";
# a question is not a statement. Its key words are jieba terms of two characters or more (three
# for Latin) that are not small talk or function words. The " and " run must start after a
# non-space: otherwise a long run of spaces is retried from every space (quadratic).
_CLAUSE_END = re.compile(
    r"([，。；;！!？?\n]|[,.](?=\s|$)|(?<=\S)\s+(?:and|but)\s+)", re.IGNORECASE
)
_SENTENCE = re.compile(r"[^。！？!?\n]+?(?:[。！？!?]+|\.(?=\s|$)|\n|$)")
MAX_STATEMENT_CHARS = 200  # a statement sent back in LEFT_OUT
_NOT_KEYS = frozenset(
    (
        "你好 您好 对了 顺便 一下 一家 一个 一些 最近 现在 今天 明天 昨天 "
        "平时 我们 你们 他们 她们 自己 这个 那个 因为 所以 但是 然后 还是 "
        "已经 可以 就是 什么 怎么 怎么样 谢谢 一直 还有 另外 其实 觉得 知道 "
        "开始 帮我 记一下 跟你说 说一下 "
        "the and but for with have has had been was were are also just really actually very "
        "about that this they them their there then than what when where how who will would "
        "could should can our your you his her she him from into lot lots some any all thanks "
        "thank way by i'm i've i'll i'd it's that's okay yes yeah hello"
    ).split()
)

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
        # A statement none of whose key words reaches a fact is asked for the same way ("在一家银行
        # 做会计" was dropped whole, 2026-11-21); one that names a left-out name is that name's.
        recheck = self._left_out_prompt
        missing = _missing(spoken, facts) if recheck is not None else []
        if recheck is not None and missing:
            # Only the sentences that hold what was left out: given the whole message again,
            # qwen3:8b dropped the job clause again (8/8); given that sentence alone it kept the
            # job (4/4), and with this second ask 8/8. A sentence that leans on an earlier one
            # ("她是医生") loses who it is about: accepted, measured on the eval sets.
            asked = [m.item for m in missing]
            focus = [{"role": "user", "content": _holding(said, asked)}]
            again, provenance = self._llm.structured(  # same role and model: one tag for both
                "extract",
                self._prompt + recheck,
                {**payload, "MESSAGES": focus, "LEFT_OUT": asked},
                ExtractedFacts,
            )
            facts = facts + [f for f in again.facts if _covers(f, missing)]
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
    # walk back over the spaces, never copy the text before `at`: that made `mentioned` quadratic
    while at > 0 and text[at - 1].isspace():
        at -= 1
    return at == 0 or text[at - 1] in _SENTENCE_ENDS


def statements(text: str) -> Iterator[tuple[str, frozenset[str]]]:
    """The statements `text` makes that carry key words, each with them (see `_CLAUSE_END`)."""
    parts = _CLAUSE_END.split(text)  # clause, end, clause, end, ..., clause
    for clause, end in zip(parts[::2], [*parts[1::2], ""], strict=True):
        clause = clause.strip()
        if not clause or end in ("?", "？"):
            continue
        keys = frozenset(t for t in terms(clause) if _key(t))
        if keys:
            yield clause[:MAX_STATEMENT_CHARS], keys


def _key(term: str) -> bool:
    return len(term) >= (2 if _CJK.search(term) else 3) and term not in _NOT_KEYS


def _holding(said: list[dict[str, str]], asked: list[str]) -> str:
    """The user's sentences that hold one of `asked`, in order: whole sentences, so a clause keeps
    who it is about."""
    return " ".join(
        sentence.strip()
        for turn in said
        if turn["role"] == "user"
        for sentence in _SENTENCE.findall(turn["content"])
        if any(item in sentence for item in asked)
    )


class LeftOut(NamedTuple):
    item: str  # what goes into LEFT_OUT
    keys: frozenset[str]  # a fact that holds one of them covers it


def _missing(text: str, facts: list[ExtractedFact]) -> list[LeftOut]:
    """What `text` says that no fact covers, at most MAX_LEFT_OUT: names and places first, then the
    statements whose own words (names aside: a name covers only itself) reach no fact and that hold
    no left-out name (it asks for them). Stops at the cap: the text can be 200 x 20,000 chars."""
    told = " ".join(f.content for f in facts).lower()
    names = mentioned(text)
    named = frozenset(n.lower() for n in names)
    found = [LeftOut(n, frozenset({n.lower()})) for n in names if n.lower() not in told]
    found = found[:MAX_LEFT_OUT]
    asked = [m.item for m in found]
    for statement, keys in statements(text):
        if len(found) == MAX_LEFT_OUT:
            break
        own = keys - named
        if (
            own
            and not any(k in told for k in own)
            and not any(n in statement for n in asked)
            and not find(statement)  # never ask the model to write a secret down
        ):
            found.append(LeftOut(statement, own))
    return found


def _covers(fact: ExtractedFact, missing: list[LeftOut]) -> bool:
    content = fact.content.lower()
    return any(k in content for m in missing for k in m.keys)
