"""Evaluation (design §8): the same synthetic conversations through several memory systems —
lite-mem configurations and mem0 — scored without an LLM judge.

After each session, `present` facts must be among the user's current memories (any of their
keywords, any language), `stale` ones must not be (a current memory with an `old` keyword and none
of the `unless` keywords, which put the old value in the past: "moved from", "以前"). After
the last session each question is searched; a hit is a top-k memory containing any answer
keyword. Write latency and the number of memories are reported too."""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel

from lite_mem.config import Settings
from lite_mem.engine import MemoryEngine
from lite_mem.errors import LiteMemError
from lite_mem.types import Message, Scope


class Stale(BaseModel):
    old: list[str]
    unless: list[str] = []


class Session(BaseModel):
    messages: list[dict[str, str]]
    present: list[list[str]] = []
    stale: list[Stale] = []


class Question(BaseModel):
    q: str
    answers: list[str]


class Case(BaseModel):
    user_id: str
    sessions: list[Session]
    questions: list[Question] = []


class Dataset(BaseModel):
    users: list[Case]


def load(path: Path) -> Dataset:
    return Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _has(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def remembered(memories: list[str], keywords: list[str]) -> bool:
    return any(_has(m, keywords) for m in memories)


def stale(memories: list[str], s: Stale) -> bool:
    return any(_has(m, s.old) and not _has(m, s.unless) for m in memories)


class System(Protocol):
    """A memory system under evaluation (lite-mem in one configuration, or mem0)."""

    name: str

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None: ...

    def current(self, user_id: str) -> list[str]: ...

    def search(self, user_id: str, query: str, k: int) -> list[str]: ...

    def reset(self, user_ids: list[str]) -> None: ...

    def maintain(self, user_id: str) -> int:
        """Offline consolidation after a session (S7); how many memories it closed."""
        ...


class LiteMemSystem:
    def __init__(self, name: str, engine: MemoryEngine, *, maintain: bool = False) -> None:
        self.name = name
        self._engine = engine
        self._maintain = maintain

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None:
        said = [Message(m["role"], m["content"]) for m in messages]
        self._engine.add(said, Scope(user_id=user_id))

    def current(self, user_id: str) -> list[str]:
        return [m.content for m in self._engine.list_memories(Scope(user_id=user_id))]

    def search(self, user_id: str, query: str, k: int) -> list[str]:
        hits = self._engine.search(query, Scope(user_id=user_id), top_k=k)
        return [h.memory.content for h in hits]

    def reset(self, user_ids: list[str]) -> None:
        for user_id in user_ids:
            self._engine.erase(user_id)

    def maintain(self, user_id: str) -> int:
        if not self._maintain:
            return 0
        return self._engine.maintain(user_id).closed


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


@dataclass
class Report:
    system: str
    k: int = 3  # hit@k
    present: int = 0  # checks of facts that must be remembered
    remembered: int = 0
    stale_checks: int = 0
    stale: int = 0  # facts still current that should not be
    questions: int = 0
    hits: int = 0
    memories: int = 0  # current memories at the end, all users
    closed: int = 0  # memories maintenance closed (expired, merged, summarised, forgotten)
    errors: int = 0  # calls that failed (the session or question still counts, as a miss)
    add_seconds: list[float] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)  # what went wrong, for reading

    @property
    def recall(self) -> float:
        return _ratio(self.remembered, self.present)

    @property
    def stale_rate(self) -> float:
        return _ratio(self.stale, self.stale_checks)

    @property
    def hit_rate(self) -> float:
        return _ratio(self.hits, self.questions)

    @property
    def seconds_per_add(self) -> float:
        return sum(self.add_seconds) / len(self.add_seconds) if self.add_seconds else 0.0


def run(system: System, dataset: Dataset, *, k: int = 3) -> Report:
    report = Report(system.name, k)
    system.reset([u.user_id for u in dataset.users])
    for case in dataset.users:
        now: list[str] = []
        for n, session in enumerate(case.sessions, start=1):
            started = time.perf_counter()
            try:  # one failed model call (a local endpoint's 500) costs a session, not the run
                system.add(case.user_id, session.messages)
                report.closed += system.maintain(case.user_id)
            except LiteMemError as exc:
                report.errors += 1
                report.misses.append(f"{case.user_id} s{n}: {exc}")
            report.add_seconds.append(time.perf_counter() - started)
            now = system.current(case.user_id)
            for keywords in session.present:
                report.present += 1
                if remembered(now, keywords):
                    report.remembered += 1
                else:
                    report.misses.append(f"{case.user_id} s{n}: forgot {keywords[0]!r}")
            for s in session.stale:
                report.stale_checks += 1
                if stale(now, s):
                    report.stale += 1
                    report.misses.append(f"{case.user_id} s{n}: still remembers {s.old[0]!r}")
        for question in case.questions:
            report.questions += 1
            try:
                found = system.search(case.user_id, question.q, k)
            except LiteMemError as exc:
                report.errors += 1
                report.misses.append(f"{case.user_id}: {question.q!r}: {exc}")
                continue
            if remembered(found, question.answers):
                report.hits += 1
            else:
                report.misses.append(f"{case.user_id}: no hit for {question.q!r}")
        report.memories += len(now)  # after the last session (searching changes nothing)
    return report


def table(reports: list[Report]) -> str:
    """A markdown comparison table."""
    k = reports[0].k if reports else 3
    lines = [
        f"| system | remembered | stale | hit@{k} | memories | closed | errors | s/add |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        lines.append(
            f"| {r.system} | {r.remembered}/{r.present} ({r.recall:.0%}) "
            f"| {r.stale}/{r.stale_checks} ({r.stale_rate:.0%}) "
            f"| {r.hits}/{r.questions} ({r.hit_rate:.0%}) | {r.memories} | {r.closed} "
            f"| {r.errors} "
            f"| {r.seconds_per_add:.1f} |"
        )
    return "\n".join(lines)


def describe(settings: Settings, sections: set[str]) -> dict[str, Any]:
    """The settings a run used, safe to save: keys masked (pydantic), endpoints cut to scheme,
    host and path (no user info or query string, where a token can hide)."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _endpoint(v) if k == "base_url" else clean(v) for k, v in value.items()}
        return value

    return clean(settings.model_dump(mode="json", include=sections))  # type: ignore[no-any-return]


def _endpoint(url: str | None) -> str | None:
    if not url:
        return url
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}{parts.path}"


def save(path: Path, reports: list[Report], *, dataset: Path, settings: dict[str, Any]) -> None:
    """One run as JSON, with what produced it, so runs can be compared later (S7 sweeps)."""
    try:
        git = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    rows = [
        {**asdict(r), "recall": r.recall, "stale_rate": r.stale_rate, "hit_rate": r.hit_rate}
        for r in reports
    ]
    record = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git,
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "k": reports[0].k if reports else None,
        "settings": settings,
        "reports": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


# Built-in lite-mem variants (Settings overrides on top of the environment's). "thinking" leaves
# qwen3's reasoning on: about 20x slower per call on this machine.
@dataclass(frozen=True)
class Variant:
    overrides: dict[str, Any]  # Settings sections, merged over the environment's
    maintain: bool = False  # run `maintain` after each session


_NO_THINKING = {"llm": {"reasoning": "none"}}
_TERSE = {**_NO_THINKING, "pipeline": {"extract_prompt": "extract_facts_terse.j2"}}
VARIANTS: dict[str, Variant] = {
    "resolve": Variant(_NO_THINKING),
    "add_only": Variant({**_NO_THINKING, "pipeline": {"resolver": "add_only"}}),
    "terse_prompt": Variant(_TERSE),
    "thinking": Variant({}),
    # S7: the terse variant with maintenance after every session, one merge threshold each.
    **{
        f"merge_{round(t * 100)}": Variant(
            {**_TERSE, "maintain": {"merge_threshold": t}}, maintain=True
        )
        for t in (0.80, 0.85, 0.90, 0.95)
    },
}


def variant_settings(base: Settings, name: str, schema: str) -> Settings:
    """`base` with the variant's overrides, its own namespace and the eval schema."""
    data = base.model_dump()
    for section, values in VARIANTS[name].overrides.items():
        data[section] = {**data[section], **values}
    data.update(namespace=f"eval/{name.replace('_', '-')}", db_schema=schema)
    return Settings.model_validate(data)
