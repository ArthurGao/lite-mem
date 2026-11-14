"""lite-mem as a lite-eval target (design §8, S11): the two-hop eval and LongMemEval on lite-eval.

Both store memories as said (`infer=False`: no model call, one memory per user message) in a
throwaway SQLite database, with the configured embedder (bge-m3 on the local Ollama), and search
with `top_k` and `follow_up` from the suite's config.

- `multihop`: the S10 eval (`eval/lite-eval/suites/multihop.yaml`). setup stores each user's
  memories (`corpus/<user>.json`) once for every config of the suite (they share it, lite-eval
  §18.18); a case is one question in its user's scope, its output the hits' contents, best first,
  scored by lite-eval's memory pack (`hop1`, `hop2`, `all_hops`: the old `eval/multihop.py` rule).
  BM25 statistics are the scope's (S8b), so storing every user first finds what the old script
  found storing and asking user by user.
- `longmemeval`: session recall on LongMemEval (`eval/longmemeval.py`). Each case brings its own
  history (`corpus/<question>.json`): a fresh database per case gets the user turns of every
  session (assistant turns are not memories; see `MemoryEngine.add`), each tagged with its
  session id; the question's hits become session ids, best first, and lite-eval's `recall@k` /
  `hit@k` score them (repeated ids count once). Only ids leave: run records keep no benchmark
  text. Each question's database is built once and searched by every config (they differ only in
  top_k and follow_up); the databases live in the run's temp dir until the run ends (M: several
  GB of disk). Vectors are cached by text for the whole run: S's 122k user turns hold 94k distinct
  texts, and bge-m3 on the local Ollama embeds ~38 a second.
"""

import asyncio
import json
import shutil
import tempfile
import threading
from array import array
from dataclasses import dataclass, field, replace
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from lite_eval.contracts import Case, Output, Span, Trace
from pydantic import JsonValue

from lite_mem import __version__
from lite_mem.config import Settings
from lite_mem.embedding import Embedder
from lite_mem.engine import MAX_FACT_CHARS, MAX_MESSAGES, MemoryEngine
from lite_mem.factory import Components, open_components, open_store
from lite_mem.types import Message, Scope

if TYPE_CHECKING:
    from lite_eval.datasets.local import Dataset

NAMESPACE = "eval"
_USER = Scope(user_id="user")  # each LongMemEval question is one user's history


def _settings(db: Path) -> Settings:
    return Settings(database_url=f"sqlite:///{db}", namespace=NAMESPACE)


class _LiteMemTarget:
    name = "lite-mem"

    def __init__(self, top_k: int, follow_up: int = 0) -> None:
        self.top_k = top_k
        self.follow_up = follow_up
        self.embedding = Settings().embedding  # what every engine of this target embeds with
        self._dir: Path | None = None  # the databases; only the setup's owner (leader) has it

    def _embedding_identity(self) -> dict[str, Any]:
        """Every embedding setting that decides a vector (not the key, not the timeout)."""
        return self.embedding.model_dump(mode="json", include={"provider", "model", "base_url"})

    def config(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "follow_up": self.follow_up,
            "store": "sqlite",
            "embedding": self._embedding_identity(),
        }

    def versions(self) -> dict[str, str]:
        return {"lite-mem": __version__}

    def info(self) -> dict[str, str]:
        return {}

    def billing(self) -> Literal["free", "per_token", "unknown"]:
        return "free" if self.embedding.provider == "ollama" else "per_token"

    def setup_identity(self) -> dict[str, Any]:
        return {"embedding": self._embedding_identity()}

    def _make_dir(self) -> Path:
        self._dir = Path(tempfile.mkdtemp(prefix="lite-mem-eval-"))
        return self._dir

    def _remove_dir(self) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None


class MultihopTarget(_LiteMemTarget):
    """The S10 two-hop eval; see the module docstring."""

    def __init__(self, top_k: int, follow_up: int = 0) -> None:
        super().__init__(top_k, follow_up)
        self.engine: MemoryEngine | None = None

    async def setup(self, dataset: "Dataset") -> None:
        db = self._make_dir() / "multihop.db"
        try:
            engine = self.engine = open_components(_settings(db)).engine(NAMESPACE)
            for path in sorted(dataset.corpus_dir.glob("*.json")):  # one file per user
                scope = Scope(user_id=path.stem)
                for memory in json.loads(path.read_text(encoding="utf-8")):
                    engine.add([Message("user", memory)], scope, infer=False)
        except BaseException:
            await self.teardown()  # lite-eval tears down only targets whose setup succeeded
            raise

    async def share_setup_from(self, leader: "MultihopTarget") -> None:
        self.engine = leader.engine

    async def run(self, case: Case) -> Output:
        assert self.engine is not None and isinstance(case.input, dict)
        question, user = str(case.input["question"]), str(case.input["user"])
        hits = await asyncio.to_thread(
            self.engine.search,
            question,
            Scope(user_id=user),
            top_k=self.top_k,
            follow_up=self.follow_up,
        )
        return Output(case_id=case.id, value=[h.memory.content for h in hits])

    async def teardown(self) -> None:
        if self._dir is not None and self.engine is not None:  # the leader's engine
            self.engine.close()
        self.engine = None
        self._remove_dir()


class CachedEmbedder:
    """The configured embedder, each distinct text embedded once (exact vectors: doubles)."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._vectors: dict[str, array[float]] = {}
        self._lock = threading.Lock()

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    def warm(self, texts: list[str]) -> None:
        """Embed what is not cached yet. The model is called outside the lock, so other cases
        build and search meanwhile (two cases may then embed a shared text twice: harmless)."""
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if missing:
            vectors = self._inner.embed(missing)
            with self._lock:
                for text, vector in zip(missing, vectors, strict=True):
                    self._vectors.setdefault(text, array("d", vector))

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.warm(texts)
        return [self._vectors[t].tolist() for t in texts]


@dataclass
class _Haystacks:
    """What every config of a LongMemEval run shares (lite-eval §18.18): the databases, built
    once per question by the first config that needs it, the embedding cache, the engine's parts
    (all but the store, which is per question) and what each build stored."""

    dir: Path
    corpus: Path
    embedder: CachedEmbedder
    components: Components
    built: dict[str, dict[str, JsonValue]] = field(default_factory=dict)  # store span attrs


class LongMemEvalTarget(_LiteMemTarget):
    """LongMemEval session recall; see the module docstring."""

    def __init__(self, top_k: int, follow_up: int = 0) -> None:
        super().__init__(top_k, follow_up)
        self._shared: _Haystacks | None = None

    async def setup(self, dataset: "Dataset") -> None:
        root = self._make_dir()
        try:
            components = open_components(_settings(root / "placeholder.db"))
        except BaseException:
            self._remove_dir()  # lite-eval tears down only targets whose setup succeeded
            raise
        components.store.close()  # every question opens its own
        embedder = CachedEmbedder(components.embedder)
        self._shared = _Haystacks(
            root, dataset.corpus_dir, embedder, replace(components, embedder=embedder)
        )

    async def share_setup_from(self, leader: "LongMemEvalTarget") -> None:
        """The leader's databases and cache: only top_k and follow_up differ between configs."""
        self._shared = leader._shared

    async def run(self, case: Case) -> Output:
        return await asyncio.to_thread(self._run, case)

    def _engine(self, db: Path) -> MemoryEngine:
        assert self._shared is not None
        return replace(self._shared.components, store=open_store(_settings(db))).engine(NAMESPACE)

    def _build(self, db: Path, haystack: str) -> dict[str, JsonValue]:
        """The question's history as memories: its user turns, tagged with their session."""
        assert self._shared is not None
        path = self._shared.corpus / f"{haystack}.json"
        sessions = json.loads(path.read_text(encoding="utf-8"))
        turns = [
            (str(s["id"]), t["content"])
            for s in sessions
            for t in s["turns"]
            if t["role"] == "user" and t["content"].strip()
        ]
        # A memory is at most MAX_FACT_CHARS (stripped, as the engine counts): the engine refuses
        # a write holding a longer message, all of it. Such a turn is left out, and counted.
        kept = [(sid, text) for sid, text in turns if len(text.strip()) <= MAX_FACT_CHARS]
        self._shared.embedder.warm([text for _, text in kept])  # one model call, not one a session
        by_session: dict[str, list[str]] = {}
        for sid, text in kept:
            by_session.setdefault(sid, []).append(text)
        engine: MemoryEngine | None = None
        stored = 0
        try:
            engine = self._engine(db)
            for sid, texts in by_session.items():
                for chunk in batched(texts, MAX_MESSAGES):
                    messages = [Message("user", t) for t in chunk]
                    meta = {"session_id": sid}
                    stored += len(engine.add(messages, _USER, infer=False, metadata=meta))
        except BaseException:  # nothing half-built is left for the next config to build on
            if engine is not None:
                engine.close()
            for part in db.parent.glob(f"{db.name}*"):  # with its -wal and -shm
                part.unlink(missing_ok=True)
            raise
        engine.close()
        return {"turns": len(turns), "stored": stored, "sessions": len(sessions)}

    def _run(self, case: Case) -> Output:
        assert self._shared is not None and isinstance(case.input, dict)
        haystack = str(case.input["haystack"])  # a corpus file name: safe as a file name
        db = self._shared.dir / f"{haystack}.db"
        if haystack not in self._shared.built:
            self._shared.built[haystack] = self._build(db, haystack)
        engine = self._engine(db)
        try:
            question = str(case.input["question"])
            hits = engine.search(question, _USER, top_k=self.top_k, follow_up=self.follow_up)
        finally:
            engine.close()
        store = Span(kind="store", name="user turns", attrs=self._shared.built[haystack])
        return Output(
            case_id=case.id,
            value=[{"id": h.memory.metadata["session_id"]} for h in hits],
            trace=Trace(spans=[store]),
        )

    async def teardown(self) -> None:
        self._shared = None
        self._remove_dir()


def multihop(top_k: int, follow_up: int = 0) -> MultihopTarget:
    return MultihopTarget(top_k, follow_up)


def longmemeval(top_k: int, follow_up: int = 0) -> LongMemEvalTarget:
    return LongMemEvalTarget(top_k, follow_up)
